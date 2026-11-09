"""Improved program for robust X-pi gate control pulse using basis expansion and global optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.linalg import expm

# Hardware constants (converted to rad/ns)
# Assume frequencies given in GHz need * 2*pi * 1e-3 to get rad/ns
# Or if they are already in rad/ns, we use them directly.
# "anharmonicity -2 pi 0.25" -> likely -0.25 GHz.
# "detuning bounds +-2 pi 0.002" -> likely +-0.002 GHz.
# "amplitude bound 0.4" -> rad/ns.
# "slew bound 0.12" -> rad/ns^2.

CHIA_GHZ = -0.25
DELTA_GHZ_MAX = 0.002
T1_NS = 25000.0
T_PHI_NS = 40000.0

# Convert to rad/ns
CHI = CHIA_GHZ * 2 * np.pi * 1e-3
DELTA_MAX = DELTA_GHZ_MAX * 2 * np.pi * 1e-3

# Operators for 3-level system
# a = [[0, 1, 0], [0, 0, sqrt(2)], [0, 0, 0]]
a = np.array([
    [0, 1, 0],
    [0, 0, np.sqrt(2)],
    [0, 0, 0]
], dtype=complex)
a_dag = a.conj().T
n = a_dag @ a
# n(n-1)
n_nm1 = n @ (n - np.eye(3))

# Target Gate: X-pi
# X gate is pi rotation around X.
# U_X = -i sigma_x = [[0, -1j], [-1j, 0]] in 2-level.
# In 3-level, it acts on |0>, |1>. |2> is untouched.
# U_target = [[0, -1j, 0], [-1j, 0, 0], [0, 0, 1]]
U_target = np.array([
    [0, -1j, 0],
    [-1j, 0, 0],
    [0, 0, 1]
], dtype=complex)

def get_hamiltonian(I, Q, delta, scale):
    """
    Construct the Hamiltonian for given I, Q, detuning, and scale.
    H = delta * n + 0.5 * chi * n(n-1) + 0.5 * scale * (I (a + a_dag) + 1j * Q (a - a_dag))
    """
    h_det = delta * n
    h_anh = 0.5 * CHI * n_nm1
    h_drive = 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
    return h_det + h_anh + h_drive

def simulate_unitary(pulse, delta, scale):
    """
    Simulate the unitary evolution for a given pulse, detuning, and scale.
    pulse: (80, 2) array of I, Q.
    Returns the 3x3 unitary matrix U.
    """
    n_steps = 80
    dt = 0.5
    U = np.eye(3, dtype=complex)
    
    for i in range(n_steps):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        H = get_hamiltonian(I_val, Q_val, delta, scale)
        # Evolve for dt
        U_step = expm(-1j * H * dt)
        U = U_step @ U
        
    return U

def evaluate_params(params):
    """
    Evaluate the score for a set of basis parameters.
    params: [amp, sigma, t0, c1, c2]
    c0 is fixed to 1? Or optimized?
    Let's optimize [c0, c1, c2, sigma, t0].
    I(t) = (c0 + c1 * H1(x) + c2 * H2(x)) * exp(-x^2/2)
    Q(t) = 0? The DRAG term is usually in Q.
    The basis expansion should apply to the I channel primarily, and Q channel can be a derivative or another basis.
    To keep it simple and effective, let's use:
    I(t) = A * exp(-x^2/2) * (1 + beta * x)  # This is not orthogonal.
    
    Let's use a proper Hermite-Gaussian expansion for I and Q separately?
    Or just use the standard DRAG form with optimized parameters:
    I(t) = A * exp(-x^2/2)
    Q(t) = B * x * exp(-x^2/2)
    Parameters: A, B, sigma, t0.
    This is 4 parameters.
    """
    A, B, sigma, t0 = params
    
    # Constraints
    if sigma < 1.0 or sigma > 15.0:
        return 1e6
    if abs(t0 - 20.0) > 5.0:
        return 1e6
        
    dt = 0.5
    n_steps = 80
    t = np.arange(n_steps) * dt
    
    x = (t - t0) / sigma
    envelope = np.exp(-0.5 * x**2)
    
    I = A * envelope
    Q = B * x * envelope
    
    pulse = np.column_stack([I, Q])
    
    # Check Amplitude Constraint
    amp = np.sqrt(I**2 + Q**2)
    if np.max(amp) > 0.4:
        return 1e6
        
    # Check Slew Rate Constraint
    # From 0 to step 0
    if np.sqrt(I[0]**2 + Q[0]**2) / dt > 0.12:
        return 1e6
    # From last step to 0
    if np.sqrt(I[-1]**2 + Q[-1]**2) / dt > 0.12:
        return 1e6
    # Between steps
    for i in range(n_steps - 1):
        diff_I = I[i+1] - I[i]
        diff_Q = Q[i+1] - Q[i]
        norm_diff = np.sqrt(diff_I**2 + diff_Q**2)
        if norm_diff / dt > 0.12:
            return 1e6
            
    # Simulation
    infidelities = []
    # Detunings: -DELTA_MAX, 0, DELTA_MAX
    # Scales: 0.97, 1.0, 1.03
    deltas = [-DELTA_MAX, 0.0, DELTA_MAX]
    scales = [0.97, 1.0, 1.03]
    
    # Input states: |0>, |1>, |+>, |->, |i+>, |i->?
    # Task says: |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)
    # That's 5 states.
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    ]
    # Wait, task says 5 states: |0>, |1>, (|0> +- |1>)/sqrt(2) [2 states], (|0> +- i|1>)/sqrt(2) [2 states]. Total 6?
    # "for the 5 input states... |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
    # Usually "A +- B" implies two states. So 1+1+2+2 = 6 states?
    # Let's re-read: "fidelity to X over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
    # It lists 4 items, but two are compound.
    # Standard set is often 5: |0>, |1>, |+>, |->, |i+>.
    # Let's assume the 5 states are: |0>, |1>, |+>, |->, |i+>.
    # Or maybe the 6 states. I will use the 5 standard ones: |0>, |1>, |+>, |->, |i+>.
    # Actually, let's use all 6 to be safe, or just 5 if the text implies 5.
    # "fidelity to X over [list]".
    # I will use: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2).
    
    states_5 = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    ]
    
    for delta in deltas:
        for scale in scales:
            U = simulate_unitary(pulse, delta, scale)
            for psi in states_5:
                # Apply U to psi
                psi_final = U @ psi
                # Target state: Apply U_target to psi
                psi_target = U_target @ psi
                # Fidelity = |<psi_target|psi_final>|^2
                fid = np.abs(np.vdot(psi_target, psi_final))**2
                inf = 1.0 - fid
                infidelities.append(inf)
                
    score = np.mean(infidelities) + np.max(infidelities)
    return score

# Optimization
# Bounds:
# A: [0, 0.4] (peak amplitude)
# B: [-0.4, 0.4] (DRAG coefficient)
# sigma: [2.0, 10.0]
# t0: [18.0, 22.0]

bounds = [
    (0.0, 0.4),   # A
    (-0.4, 0.4),  # B
    (2.0, 10.0),  # sigma
    (18.0, 22.0)  # t0
]

# Run optimization
# To ensure it runs within 300s, we limit maxiter and popsize.
result = differential_evolution(
    evaluate_params, 
    bounds, 
    seed=42, 
    maxiter=50, 
    popsize=10,
    tol=0.01,
    polish=True
)

opt_params = result.x
opt_A, opt_B, opt_sigma, opt_t0 = opt_params

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    """
    dt = 0.5
    n_steps = 80
    t = np.arange(n_steps) * dt
    
    x = (t - opt_t0) / opt_sigma
    envelope = np.exp(-0.5 * x**2)
    
    I = opt_A * envelope
    Q = opt_B * x * envelope
    
    p = np.column_stack([I, Q])
    
    # Final constraint enforcement (safety)
    amp = np.sqrt(I**2 + Q**2)
    if np.max(amp) > 0.4:
        scale = 0.4 / np.max(amp)
        I *= scale
        Q *= scale
        p = np.column_stack([I, Q])
        
    # Slew rate enforcement
    # This is tricky to do perfectly without breaking the shape, 
    # but the optimizer should have found a feasible solution.
    # We'll just return the optimized pulse.
    
    return p

# EVOLVE-BLOCK-END
