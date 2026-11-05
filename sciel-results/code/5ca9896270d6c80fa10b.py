"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for fast operations
    A = np.array(adjacency, dtype=np.float64)
    
    # Deadline guard: The evaluator kills after 150s. 
    # We aim to finish well before that to be safe, say 145s.
    start_time = time.time()
    deadline = 145.0
    
    best_labels = None
    best_cut = -1
    
    # Helper to calculate cut size
    def calc_cut_size(labels):
        # labels are 0/1. 
        # Cut size = 0.5 * sum_{i,j} A[i][j] * (labels[i] != labels[j])
        mask = labels[:, None] != labels[None, :]
        return 0.5 * np.sum(A * mask)
    
    # Quantum Annealing / Spin Glass Simulation
    # We model the system as a spin glass with a transverse field.
    # State: complex vector z of length n.
    # Hamiltonian: H = -0.5 * z^H A z - g(t) * sum |z_i|^2 (approx)
    # Actually, standard QAOA/Quantum Annealing analogy for classical optimization:
    # We evolve a state vector in a high-dimensional space or use a stochastic spin update
    # influenced by a "quantum" term that allows tunneling.
    
    # A simpler, effective "Quantum Annealing" surrogate for CPU:
    # Use a complex spin vector x.
    # Update rule based on gradient of the energy with respect to the angle of the spin.
    # Energy E(x) = - x^T A x (for real spins, max cut is min -x^T A x).
    # For complex spins, we can use a variational approach.
    
    # However, a more direct "Quantum Annealing" simulation often used in heuristics
    # involves a temperature-like schedule and a "tunneling" probability.
    
    # Let's implement a vectorized version of the "Quantum Annealing" heuristic
    # often described in literature for MaxCut:
    # 1. Start with a random complex state.
    # 2. Define a schedule for the transverse field strength gamma(t).
    # 3. Update spins based on the local field and the transverse field.
    
    # For n=100, we can afford multiple restarts.
    
    def quantum_annealing_init():
        # Initialize with a random spin state (real part dominant, small imaginary part)
        # Or start with a spectral initialization if we want, but the prompt asks for a different strategy.
        # Let's stick to random complex initialization for the "Quantum" flavor, 
        # but maybe mix in spectral for better performance? 
        # The directive says "different design". Let's use random complex.
        real_part = np.random.randn(n)
        imag_part = np.random.randn(n) * 0.1 # Small imaginary part initially
        z = real_part + 1j * imag_part
        return z
    
    def run_quantum_annealing(z_init, max_time=5.0):
        # Parameters
        t_start = time.time()
        
        # Schedule
        # gamma(t) is the transverse field. Starts high, goes to 0.
        # We use a linear schedule for simplicity, or cosine.
        # Let's use a fixed number of steps and check time.
        
        # State
        z = z_init.copy()
        
        # We want to maximize the cut.
        # Let's define the energy to minimize: E = -0.25 * sum_{i,j} A[i,j] (1 - x_i x_j) 
        # For real spins x_i = Re(z_i) / |z_i|? Or just project to real?
        
        # A common heuristic "Quantum Annealing" for classical problems:
        # Treat the state as a probability distribution over configurations? No, too slow.
        # Treat it as a continuous optimization problem with a complex gradient flow?
        
        # Let's use the "Quantum-inspired" update:
        # The local field for spin i is h_i = sum_j A[i,j] x_j.
        # The update is driven by h_i and a random "quantum" noise term scaled by gamma.
        
        # To make it "Quantum", we allow the spins to have a phase.
        # But MaxCut is a real problem. 
        # Let's use the formulation where we minimize x^T A x with x in {+1, -1}.
        # We relax x to be a continuous vector, but add a "quantum" term that allows 
        # crossing the energy barrier.
        
        # Actually, a very effective method that fits this description is 
        # **Simulated Quantum Annealing** (SQA).
        # SQA uses a transverse field to allow tunneling.
        # The state is a vector of spins. The update is stochastic.
        
        # Vectorized SQA step:
        # 1. Calculate local fields h = A @ x
        # 2. Calculate the "force" F = h
        # 3. The probability of flipping spin i is determined by:
        #    P_flip ~ sigmoid( (h_i * x_i - gamma * noise) / T )? 
        #    Actually, in SQA, the effective field is h_i, and the transverse field 
        #    adds a random component.
        
        # Let's implement a simplified vectorized SQA:
        # x is the spin vector (+1/-1).
        # At each step:
        # 1. Compute h = A @ x
        # 2. Compute the "energy gain" if we flip i: Delta E_i = -2 * h_i * x_i
        # 3. In classical SA, we flip if Delta E < 0 or with prob exp(-Delta E / T).
        # 4. In SQA, the transverse field allows flipping even if Delta E > 0, 
        #    effectively tunneling through barriers.
        #    The probability is modified by the transverse field strength gamma.
        
        # Let's use a specific SQA update rule:
        # The effective field is h_i.
        # We generate a random number r_i in [0, 1].
        # We flip i if r_i < sigmoid( (2 * h_i * x_i) / (T + gamma^2) )? 
        # This is a bit hand-wavy. Let's use a more standard implementation.
        
        # Standard SQA:
        # The Hamiltonian is H = - sum h_i s_i - gamma sum s_i^x
        # The state is a superposition. But we are doing classical simulation of the quantum system.
        
        # Let's use a simpler "Quantum Walk" or "Tunneling" heuristic:
        # We maintain a real spin vector x.
        # We also maintain a "momentum" or "velocity" v?
        
        # Let's go with a robust **Vectorized Simulated Annealing with Quantum Tunneling Term**.
        # The "Quantum" part is a noise term that is correlated with the gradient, 
        # allowing moves that go "uphill" in energy but "downhill" in a transformed potential.
        
        # Actually, let's look at the inspiration: "Quantum Annealing with a transverse-field schedule".
        # I will implement a discretized version of the Quantum Annealing differential equation.
        # d/dt |psi> = -i H(t) |psi>
        # For a single qubit, this rotates the state.
        # For N qubits, it's complex.
        
        # Alternative: **Quantum-Inspired Particle Swarm Optimization**? No.
        
        # Let's stick to **Simulated Quantum Annealing (SQA)** as it's a known heuristic for MaxCut.
        # Reference: "Simulated Quantum Annealing for the Max-Cut problem"
        
        # Implementation of SQA:
        # 1. Initialize x randomly.
        # 2. Set T (temperature) and gamma (transverse field).
        # 3. Loop:
        #    a. h = A @ x
        #    b. For each i, the probability of flipping is:
        #       P_i = 1 / (1 + exp( -2 * h_i * x_i / (T + gamma^2) ))
        #       Wait, if h_i * x_i is positive, the spin is aligned with the field. 
        #       We want to flip if it's misaligned?
        #       If x_i = 1 and h_i > 0, flipping to -1 increases energy (bad).
        #       If x_i = 1 and h_i < 0, flipping to -1 decreases energy (good).
        #       So we want to flip if h_i * x_i < 0.
        #       The "force" to flip is -h_i * x_i.
        #       P_flip_i = 1 / (1 + exp( 2 * h_i * x_i / (T + gamma^2) ))
        #       If h_i * x_i is very negative, P -> 1.
        #       If h_i * x_i is very positive, P -> 0.
        #       The term gamma^2 in the denominator allows flipping even if h_i * x_i > 0 (tunneling).
        
        # 4. Update x: x_i = -x_i with probability P_i.
        # 5. Cool T and gamma.
        
        x = np.random.choice([-1, 1], size=n)
        
        # Schedule
        T = 10.0
        gamma = 5.0
        cooling_rate = 0.99
        
        while time.time() - t_start < max_time:
            # Compute local fields
            h = A @ x
            
            # Compute flip probabilities
            # Note: h_i * x_i is the alignment.
            # We want to flip if alignment is negative.
            # The "effective temperature" is T + gamma^2.
            # The exponent is - (2 * h_i * x_i) / (T + gamma^2) ?
            # Let's check:
            # If h_i * x_i = -10 (misaligned), exponent = 20 / (T+g^2). P = 1/(1+exp(20/...)) -> small?
            # No, we want high probability to flip if misaligned.
            # So we want P_flip to be high when h_i * x_i is negative.
            # Let's use: P_flip = 1 / (1 + exp( 2 * h_i * x_i / (T + gamma^2) ))
            # If h_i * x_i = -10, exp(-20/...) is small, P is large. Correct.
            # If h_i * x_i = 10, exp(20/...) is large, P is small. Correct.
            
            denom = T + gamma**2
            if denom < 1e-6:
                break
            
            exponent = (2.0 * h * x) / denom
            # Clip to avoid overflow
            exponent = np.clip(exponent, -50, 50)
            p_flip = 1.0 / (1.0 + np.exp(exponent))
            
            # Stochastic update
            # Generate random numbers
            r = np.random.random(n)
            flip_mask = r < p_flip
            x[flip_mask] = -x[flip_mask]
            
            # Cooling
            T *= cooling_rate
            gamma *= cooling_rate
            
            # Check time
            if time.time() - t_start > max_time:
                break
        
        # Convert to 0/1 labels
        labels = (x > 0).astype(np.int8)
        return labels

    # Multi-restart SQA
    # We run several instances of SQA with different random seeds/inits.
    
    num_restarts = 20 # Adjust based on time
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Run SQA
        labels = run_quantum_annealing(None, max_time=5.0)
        
        # Evaluate
        cut = calc_cut_size(labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
            
        # Optional: Local search refinement (Greedy)
        # The SQA might get stuck in a local optimum that is better than random 
        # but not as good as a greedy refinement.
        # Let's add a greedy local search step to the SQA result.
        
        # Greedy Local Search
        x = labels.copy()
        x[x == 0] = -1
        x[x == 1] = 1
        
        improved = True
        while improved:
            if time.time() > deadline - 2:
                break
            h = A @ x
            gains = x * h
            max_gain_idx = np.argmax(gains)
            max_gain = gains[max_gain_idx]
            if max_gain > 0:
                x[max_gain_idx] = -x[max_gain_idx]
                improved = True
            else:
                improved = False
        
        labels_refined = (x > 0).astype(np.int8)
        cut_refined = calc_cut_size(labels_refined)
        
        if cut_refined > best_cut:
            best_cut = cut_refined
            best_labels = labels_refined.copy()

    return best_labels
# EVOLVE-BLOCK-END
