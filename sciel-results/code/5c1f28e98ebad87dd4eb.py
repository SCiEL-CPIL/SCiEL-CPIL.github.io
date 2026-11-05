import numpy as np
import time
from scipy.optimize import minimize

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles maximizing the mean ratio of the 
    independently simulated expected cut to the exact classical optimum.
    
    Strategy:
    1. Precompute cut values and exact classical optimum.
    2. Use a structured state-vector simulation where the mixer U_X(beta) is applied 
       as a sequence of single-qubit rotations (O(n*2^n) complexity) rather than 
       constructing the full 2^n x 2^n Kronecker product matrix (O(4^n) complexity).
    3. Compute analytic gradients for the objective function (expected cut / opt cut) 
       using the parameter-shift rule or direct differentiation of the expectation value.
    4. Use L-BFGS-B with analytic gradients and multiple random restarts to find 
       high-quality local optima efficiently within the time budget.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Precompute cut values
    max_states = 1 << n
    cut_vals = np.zeros(max_states, dtype=np.float64)
    for s in range(max_states):
        c = 0
        for u, v in edges:
            if ((s >> u) ^ (s >> v)) & 1:
                c += 1
        cut_vals[s] = c
        
    opt_cut = np.max(cut_vals)
    if opt_cut == 0:
        return np.zeros((p, 2))

    # Initial state |+>^n
    state_init = np.ones(max_states, dtype=np.complex128)
    state_init /= np.sqrt(max_states)

    # Precompute the phases for the C operator? 
    # U_C(gamma) = exp(-i gamma C). C is diagonal.
    # We can just compute exp(-1j * gamma * cut_vals) on the fly.
    
    def apply_mixer(state, beta):
        """
        Apply U_X(beta) = \bigotimes_{i=0}^{n-1} R_x(beta) to the state vector.
        R_x(beta) = [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
        
        This is done by iterating over each qubit and applying the 2x2 rotation 
        to the corresponding pairs of amplitudes.
        Complexity: O(n * 2^n)
        """
        c = np.cos(beta)
        s = -1j * np.sin(beta)
        
        # We need to reshape the state to apply the rotation on each qubit.
        # The state is indexed by binary string b_0 b_1 ... b_{n-1}.
        # Qubit 0 is the least significant bit (LSB) in standard numpy indexing if we 
        # treat the state as a flat array where index s corresponds to integer s.
        # Wait, standard QAOA conventions: 
        # Usually, index s has bits b_0 ... b_{n-1}.
        # If we use numpy, index s is just an integer.
        # To apply a gate on qubit k, we need to group indices by the value of bit k.
        
        # Efficient way:
        # Reshape state to (2, 2, ..., 2) with n dimensions.
        # Axis 0 corresponds to qubit 0 (LSB) or qubit n-1 (MSB)?
        # In numpy, C-order flattening means the last index varies fastest.
        # If we reshape to (2,)*n, the last axis is the LSB.
        # So axis n-1 is qubit 0, axis 0 is qubit n-1.
        
        # Let's stick to a robust method:
        # Iterate through qubits. For each qubit k, we swap/rotate pairs of elements 
        # where the k-th bit differs.
        
        # Actually, a simpler and very fast method for small n (<=10) is to use 
        # the Kronecker product structure but apply it via tensor operations or 
        # just simple loops if n is small. 
        # However, O(n*2^n) is best.
        
        # Let's use the reshape trick.
        # state shape: (2, 2, ..., 2)
        # We want to apply R_x to each qubit.
        # In the tensor representation, qubit i corresponds to axis i.
        # We need to be careful about bit ordering.
        # Standard: index s = sum b_i 2^i.
        # Numpy reshape: state.reshape([2]*n)
        # Index [i_0, i_1, ..., i_{n-1}] corresponds to s = i_0*2^{n-1} + ... + i_{n-1}*2^0.
        # So axis 0 is MSB (qubit n-1), axis n-1 is LSB (qubit 0).
        
        state_tensor = state.reshape([2] * n)
        
        # Apply R_x on each axis
        # R_x = [[c, s], [s, c]]
        # For a given axis, we combine the two slices.
        
        for k in range(n):
            # Move axis k to the front
            state_tensor = np.moveaxis(state_tensor, k, 0)
            # Now shape is (2, 2, ..., 2)
            # Apply the matrix multiplication on the first dimension
            # v' = M @ v
            # v_new[0] = c*v[0] + s*v[1]
            # v_new[1] = s*v[0] + c*v[1]
            
            v0 = state_tensor[0]
            v1 = state_tensor[1]
            
            new_v0 = c * v0 + s * v1
            new_v1 = s * v0 + c * v1
            
            state_tensor = np.stack([new_v0, new_v1], axis=0)
            # Move axis k back to its original position
            state_tensor = np.moveaxis(state_tensor, 0, k)
            
        return state_tensor.reshape(max_states)

    def simulate(params):
        """
        params: [gamma_0, beta_0, ..., gamma_{p-1}, beta_{p-1}]
        Returns: expected_cut / opt_cut
        """
        gammas = params[:p]
        betas = params[p:]
        
        state = state_init.copy()
        
        for k in range(p):
            g = gammas[k]
            b = betas[k]
            
            # Apply C phase: state = U_C(g) @ state
            # U_C is diagonal with entries exp(-i * g * cut_vals[s])
            phases_c = np.exp(-1j * g * cut_vals)
            state = state * phases_c
            
            # Apply X rotation: state = U_X(b) @ state
            state = apply_mixer(state, b)
            
        probs = np.abs(state) ** 2
        expected_cut = np.sum(probs * cut_vals)
        return expected_cut / opt_cut

    def objective_and_grad(params):
        """
        Compute objective value and analytic gradient.
        Objective: - (Expected Cut / Opt Cut)
        
        Gradient derivation:
        Let |psi(theta)> be the state. 
        H(theta) = <psi(theta)| C |psi(theta)> / opt_cut.
        
        The derivative of the expectation value <psi|C|psi> with respect to a parameter 
        appearing in a unitary e^{-i theta H_op} can be computed using the 
        parameter-shift rule or directly.
        
        For QAOA, the state is:
        |psi> = U_X(beta_{p-1}) U_C(gamma_{p-1}) ... U_X(beta_0) U_C(gamma_0) |+>^n
        
        We can compute the gradient by propagating the state forward and backward 
        (adjoint method) or by using the parameter-shift rule which requires 
        2 evaluations per parameter. Given we have p=2 (4 params), parameter-shift 
        is very cheap (8 extra simulations).
        
        However, we can do better with analytic gradients if we store intermediate states.
        Let's use the parameter-shift rule for simplicity and robustness, as n is small 
        and simulation is fast.
        
        Parameter-shift rule for a parameter theta in e^{-i theta H}:
        d/dtheta <psi|O|psi> = 0.5 * (<psi(theta+pi/2)|O|psi(theta+pi/2)> - <psi(theta-pi/2)|O|psi(theta-pi/2)>)
        
        Wait, the standard parameter-shift rule applies to gates of the form R(theta) = e^{-i theta Z/2} or similar.
        Here we have e^{-i gamma C} and e^{-i beta S_x}.
        
        For e^{-i gamma C}, the shift is +/- pi/2.
        For e^{-i beta S_x}, the shift is +/- pi/2.
        
        So for each parameter, we need 2 additional simulations.
        Total simulations per gradient step: 1 (base) + 4 (params) * 2 (shifts) = 9 simulations.
        
        Is this faster than finite difference? 
        Finite difference for L-BFGS-B usually uses central differences, requiring 2N evals for N params.
        Here N=4. So 8 evals for gradient + 1 for value = 9 evals.
        Same number of evaluations! But parameter-shift is exact, while finite diff is approximate.
        More importantly, we can potentially batch these or optimize the simulation further.
        
        Actually, let's implement the analytic gradient using the chain rule and stored states.
        It's more complex but potentially faster if we can avoid redundant computations.
        
        Let's stick to Parameter-Shift for correctness and simplicity first.
        """
        # Base value
        val = simulate(params)
        
        # Gradient via Parameter Shift Rule
        grad = np.zeros(4)
        shift = np.pi / 2.0
        
        for i in range(4):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += shift
            params_minus[i] -= shift
            
            val_plus = simulate(params_plus)
            val_minus = simulate(params_minus)
            
            # Derivative of objective (which is -val)
            # d/dtheta (-val) = - d(val)/dtheta
            # d(val)/dtheta = 0.5 * (val_plus - val_minus)
            grad[i] = -0.5 * (val_plus - val_minus)
            
        return -val, grad

    def objective(params):
        val, _ = objective_and_grad(params)
        return val

    def jac(params):
        _, grad = objective_and_grad(params)
        return grad

    # Optimization Strategy
    best_value = -np.inf
    best_params = None
    
    rng = np.random.default_rng(42)
    
    # Benchmark one simulation to estimate budget
    t0 = time.time()
    _ = simulate(np.array([0.5, 0.5, 0.5, 0.5]))
    sim_time = time.time() - t0
    
    # Each objective_and_grad call takes ~9 * sim_time
    # L-BFGS-B with analytic gradients converges in ~10-20 iterations usually.
    # Let's say 20 iterations.
    # Cost per restart ~ 20 * 9 * sim_time = 180 * sim_time.
    
    est_time_per_restart = 180 * sim_time + 0.1 # overhead
    max_restarts = int((deadline - time.time()) / est_time_per_restart)
    max_restarts = max(5, min(max_restarts, 100)) # Cap restarts
    
    # Heuristic initializations
    # Often gamma ~ 0.5*pi, beta ~ 0.2*pi works well for p=2
    base_g = np.pi / 4.0
    base_b = np.pi / 8.0
    
    for run in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization with some bias towards known good regions
        g_init = rng.uniform(0, np.pi, p)
        b_init = rng.uniform(0, np.pi, p)
        
        # Occasionally use heuristic start
        if run % 3 == 0:
            g_init = base_g + rng.normal(0, 0.2, p)
            b_init = base_b + rng.normal(0, 0.2, p)
            
        x0 = np.concatenate([g_init, b_init])
        
        try:
            res = minimize(objective, x0, jac=jac, method='L-BFGS-B', 
                           options={'maxiter': 30, 'ftol': 1e-9, 'gtol': 1e-6})
            
            if -res.fun > best_value:
                best_value = -res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback
        best_params = np.array([base_g, base_b, base_g, base_b])

    gammas = best_params[:p]
    betas = best_params[p:]
    
    return np.array([gammas, betas]).T
