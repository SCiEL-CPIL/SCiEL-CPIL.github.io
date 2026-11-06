"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def angles(n, edges, p):
    # Build the MaxCut cost matrix C.
    # C[i, j] = -1 if (i, j) is an edge, else 0.
    # The cost function for a cut S, Sbar is sum_{i in S, j in Sbar} A_{ij} = (1/4) (1^T A 1 - z^T A z)
    # where z_i = +1 if i in S, -1 if i in Sbar.
    # The operator C in the QAOA paper is often defined such that the cost is -z^T C z / 4 or similar.
    # Standard QAOA uses exp(-i gamma C).
    # Let's use the formulation: MaxCut = (1/4) sum_{i,j} A_{ij} (1 - z_i z_j) = const - (1/4) sum A_{ij} z_i z_j.
    # So the Hamiltonian H_C has diagonal elements in the Z basis.
    # H_C = - (1/4) * sum_{i,j} A_{ij} Z_i Z_j.
    # To be consistent with standard definitions where larger cost is better:
    # We want to maximize the cut value.
    # Let's define the cost function for a bitstring s (0/1) as sum_{edges (i,j)} [s_i != s_j].
    
    # We will perform a local optimization over the angles (gamma_1, beta_1, gamma_2, beta_2)
    # to maximize the expected cut value for p=2.
    # Since n is small (6, 8, 10), we can simulate the expectation value efficiently.
    # However, full state simulation is 2^n. For n=10, 2^10=1024 states. This is very fast.
    # We can use a vectorized NumPy approach to compute the expectation value for a given set of angles.

    # 1. Construct the Hamiltonians
    # H_C: Diagonal in Z basis.
    # H_B: sum X_i. In Z basis, X flips the bit.
    
    # For each state s (0 to 2^n - 1), the energy E(s) = - (1/4) * sum_{i,j in edge} A_{ij} z_i z_j
    # where z_i = 1 if bit i is 1, -1 if bit i is 0? No, usually z = (-1)^s.
    # Let's stick to the cut value directly.
    # Cut(s) = sum_{(i,j) in E} [s_i XOR s_j]
    
    # Precompute cut values for all states?
    # n=10 -> 1024 states.
    # Number of edges is small.
    # This is feasible.
    
    # However, we need to do this inside the optimization loop.
    # The optimization will call the objective function many times.
    # Precomputing the "transition amplitudes" or using a closed form might be better,
    # but for n<=10, direct simulation of the state vector is possible but might be slow if done naively in Python loops.
    # Let's use the property that H_C is diagonal and H_B is sum of X.
    # The state vector after p=2 layers:
    # |psi> = (exp(-i beta_2 H_B) exp(-i gamma_2 H_C) exp(-i beta_1 H_B) exp(-i gamma_1 H_C)) |+>^n
    
    # We can represent the state as a complex vector of length 2^n.
    # Applying H_C is just multiplying by phases.
    # Applying H_B = sum X_i is applying (exp(-i beta sum X_i)) = product_i exp(-i beta X_i).
    # exp(-i beta X) = cos(beta) I - i sin(beta) X.
    # So applying H_B is a simple operation on the state vector:
    # new_state[s] = cos(beta) * old_state[s] - 1j * sin(beta) * old_state[s ^ (1 << i)] for each i?
    # No, H_B = sum_i X_i. The operators X_i commute.
    # exp(-i beta sum X_i) = product_i (cos(beta) I - i sin(beta) X_i).
    # This is NOT the same as applying each exp(-i beta X_i) with the same beta?
    # Yes, because [X_i, X_j] = 0.
    # So we can apply the full H_B step by iterating over bits?
    # Actually, applying exp(-i beta H_B) to a state vector |psi> is:
    # (exp(-i beta H_B) |psi>)_s = sum_t <s| exp(-i beta H_B) |t> psi_t
    # Since H_B is sum X_i, in the computational basis, it flips all bits? No.
    # H_B = X_1 + X_2 + ... + X_n.
    # This operator is not diagonal in the X basis? It is diagonal in the X basis? No, X_i are diagonal in X basis.
    # So in the X basis, H_B is diagonal with eigenvalues k (number of +1s) - (n-k) = 2k-n?
    # This transformation is expensive (Walsh-Hadamard transform).
    
    # Alternative: Since n is small, we can just build the matrix for exp(-i beta H_B)?
    # Size 2^n x 2^n. For n=10, 1024x1024.
    # Building it once per angle evaluation is expensive.
    # But note: H_B is fixed for all graphs. Only the angle beta changes.
    # We can precompute the eigenvectors of H_B?
    # H_B is the same for all graphs. Its eigenvalues and eigenvectors are fixed.
    # Let's precompute the matrix M_beta for a given beta? No, beta changes.
    
    # Let's use the state vector approach with explicit application.
    # Applying exp(-i gamma H_C) is O(2^n).
    # Applying exp(-i beta H_B) ...
    # Let's observe that for p=2, we can compute the expectation value analytically or semi-analytically?
    
    # Let's try a different approach: Use the fact that the objective function is smooth and use gradient-free or gradient-based optimization.
    # To compute the expectation value for a given set of angles:
    # 1. Start with |+>^n.
    # 2. Apply exp(-i gamma_1 H_C). State is still |+>^n in X basis? No, H_C is Z-diagonal.
    #    The state becomes a superposition of Z-basis states with equal magnitude but different phases?
    #    No, |+>^n is the uniform superposition of all Z basis states.
    #    exp(-i gamma H_C) |+>^n = sum_s exp(-i gamma E(s)) / sqrt(2^n) |s>
    #    where E(s) is the cost of cut s.
    # 3. Apply exp(-i beta_1 H_B).
    #    This mixes the states.
    #    The new state coefficients c'_t = sum_s <t| exp(-i beta H_B) |s> c_s
    #    <t| exp(-i beta H_B) |s> is the matrix element.
    
    # Since n <= 10, we can precompute the matrix U_B(beta) for a grid of betas? No.
    # But we can compute the expectation value <H_C> = <psi| H_C |psi>.
    # H_C is diagonal. So <H_C> = sum_s |psi_s|^2 E(s).
    
    # Let's implement the simulation.
    # Precompute E(s) for all s.
    # For each candidate angle set (g1, b1, g2, b2):
    #   1. c0 = 1/sqrt(2^n) * ones(2^n)
    #   2. c1 = c0 * exp(-1j * g1 * E)
    #   3. c2 = apply_U_B(b1, c1)
    #   4. c3 = c2 * exp(-1j * g2 * E)
    #   5. c4 = apply_U_B(b2, c3)
    #   6. val = sum(abs(c4)^2 * E)
    
    # How to apply_U_B(beta, c)?
    # U_B = exp(-i beta sum X_i).
    # This is a product of commuting operators X_i.
    # U_B = product_i exp(-i beta X_i).
    # exp(-i beta X_i) = cos(beta) I - i sin(beta) X_i.
    # So U_B = product_i (c I - i s X_i) where c=cos(beta), s=sin(beta).
    # This is NOT simply applying each one sequentially to the vector in O(n 2^n)?
    # Let's check:
    # (A B) v = A (B v).
    # Let M_i = c I - i s X_i.
    # U_B = M_1 M_2 ... M_n.
    # We can apply this to the vector c.
    # temp = c
    # for i in range(n):
    #    new_temp = c * temp - i * s * (temp with bit i flipped)
    #    # Note: X_i flips bit i.
    #    # temp_new[s] = c * temp[s] - i * s * temp[s ^ (1 << i)]
    #    temp = new_temp
    # This is O(n 2^n). For n=10, 10 * 1024 = 10k operations. Very fast.
    
    # So the plan:
    # 1. Precompute E (cost vector) of length 2^n.
    # 2. Define objective function that takes angles (g1, b1, g2, b2) and returns -expected_cut (to minimize).
    # 3. Use scipy.optimize.minimize to find optimal angles.
    # 4. Return the angles.

    # Precomputation
    num_states = 2 ** n
    states = np.arange(num_states)
    
    # Calculate cut value for each state
    # State s is an integer. Bit i is (s >> i) & 1.
    # Cut = sum_{(i,j) in edges} [bit_i(s) != bit_j(s)]
    
    # We can vectorize this.
    # Create a matrix of bits for all states.
    # bits = np.zeros((num_states, n))
    # for i in range(n):
    #     bits[:, i] = (states >> i) & 1
    
    # This might be memory heavy if n is large, but n<=10 is fine.
    # 1024 * 10 * 8 bytes = 80 KB.
    
    # Alternatively, loop over edges.
    E = np.zeros(num_states)
    for u, v in edges:
        # Get bits u and v for all states
        # bit_u = (states >> u) & 1
        # bit_v = (states >> v) & 1
        # cut_contribution = 1 - bit_u * bit_v  (since bits are 0/1, if different, product is 0, so 1-0=1. If same, product is 1, 1-1=0)
        # Wait, if bit_u=1, bit_v=1, prod=1, contrib=0.
        # If bit_u=0, bit_v=0, prod=0, contrib=1. (But 0!=0 is False. So contrib should be 0)
        # My formula: 1 - bit_u * bit_v.
        # 0,0 -> 1. Wrong.
        # 0,1 -> 1. Correct.
        # 1,0 -> 1. Correct.
        # 1,1 -> 0. Correct.
        # So it works for 0/1 bits representing "in set" / "not in set".
        # But standard QAOA uses z_i = +/- 1.
        # Let's stick to 0/1 bits for the cut calculation.
        
        bu = (states >> u) & 1
        bv = (states >> v) & 1
        E += (bu != bv) # boolean array, sums to int
    
    # Normalize E?
    # The expected value is sum |psi_s|^2 E_s.
    # E_s is the number of cut edges.
    
    # Optimization
    def objective(params):
        g1, b1, g2, b2 = params
        # Initialize state
        c = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
        
        # Step 1: exp(-i g1 H_C)
        # H_C diagonal elements are E_s.
        # Note: The sign convention.
        # Usually H_C = - sum ... so that ground state is max cut.
        # If we want to maximize E, we should apply exp(-i gamma * (-E_matrix))?
        # Or exp(-i gamma * E_matrix) and minimize the expectation?
        # Let's assume the standard QAOA formulation where we minimize the cost of the cut (un-cut edges)?
        # No, the task says "MaxCut".
        # Let's maximize the expected cut value.
        # The phase accumulated is exp(-i gamma * E_s) if H_C has eigenvalues E_s.
        # If we want the states with high E_s to have higher probability, we need constructive interference.
        # Let's just use the formula: c_s <- c_s * exp(-1j * g * E_s)
        c = c * np.exp(-1j * g1 * E)
        
        # Step 2: exp(-i b1 H_B)
        c = apply_mixing(c, b1)
        
        # Step 3: exp(-i g2 H_C)
        c = c * np.exp(-1j * g2 * E)
        
        # Step 4: exp(-i b2 H_B)
        c = apply_mixing(c, b2)
        
        # Expectation
        probs = np.abs(c) ** 2
        val = np.sum(probs * E)
        return -val # Minimize negative value

    def apply_mixing(state, beta):
        c = np.cos(beta)
        s = np.sin(beta)
        new_state = np.zeros_like(state)
        for i in range(n):
            # Flip bit i
            # We can do this efficiently with numpy slicing or reshaping, 
            # but for n=10, a simple loop with array operations is fine.
            # state_flip = state.copy()
            # Swap state[s] and state[s ^ (1<<i)]
            # Actually, the operator is (c I - i s X_i).
            # new[s] = c * old[s] - i * s * old[s ^ (1<<i)]
            
            # We need to update in place or create a temp?
            # The operator M_i = c I - i s X_i.
            # We apply M_1, then M_2, etc.
            # So we must update the whole state vector for each i.
            
            # Efficient way:
            # Create a mask for bit i?
            # Or just loop over states? No, vectorize.
            
            # Let's use the property:
            # X_i maps |s> to |s ^ (1<<i)>.
            # So the term -i s X_i means we take the value at flipped index and multiply by -i s.
            
            # We can create a flipped copy of the current state.
            # flipped = state.copy()
            # But we need to swap pairs.
            
            # Let's do it in two parts:
            # indices where bit i is 0: idx0
            # indices where bit i is 1: idx1
            # For idx0, s' = s ^ (1<<i) is in idx1.
            # For idx1, s' = s ^ (1<<i) is in idx0.
            
            # Let's create a permutation array?
            # perm = states ^ (1 << i)
            # state_flipped = state[perm]
            
            # Then new_state = c * state - 1j * s * state_flipped
            # Wait, is this correct?
            # (c I - i s X_i) |psi> = c |psi> - i s X_i |psi>
            # The j-th component of X_i |psi> is <j| X_i |psi> = <j^ (1<<i)| psi> = psi[j ^ (1<<i)].
            # So yes, new_state = c * state - 1j * s * state[state ^ (1 << i)]
            
            flipped = state[states ^ (1 << i)]
            state = c * state - 1j * s * flipped
            
        return state

    # Initial guess
    # Use heuristics from literature or random.
    # g ~ 0.5, b ~ 0.5
    x0 = np.array([0.5, 0.5, 0.5, 0.5])
    
    # Bounds
    # Angles are periodic, but optimizer works better with bounds.
    # [0, 2*pi] is safe.
    bounds = [(0, 2 * np.pi)] * 4
    
    # Optimize
    # Use Nelder-Mead or L-BFGS-B. Nelder-Mead is robust for small dimensions.
    res = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 1000})
    
    opt_angles = res.x
    # Reshape to (p, 2)
    # Rows: (gamma_1, beta_1), (gamma_2, beta_2)
    out = np.array([
        [opt_angles[0], opt_angles[1]],
        [opt_angles[2], opt_angles[3]]
    ])
    
    return out

# EVOLVE-BLOCK-END
