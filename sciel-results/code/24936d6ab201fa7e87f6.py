import numpy as np
import time

N = 400
POP_SIZE = 40

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def get_psl(h):
        """
        Compute Peak Sidelobe Level (PSL) of the autocorrelation.
        h is an array of +/- 1.
        """
        h_pad = np.zeros(2 * N)
        h_pad[:N] = h
        H = np.fft.fft(h_pad)
        c = np.fft.ifft(H * np.conj(H)).real
        
        # Extract sidelobes
        # Lag 0 is c[0]
        # Positive lags 1..N-1 are c[1:N]
        # Negative lags -1..-(N-1) are c[2N-1] down to c[N+1]
        left = c[1:N]
        right = c[N+1:2*N]
        
        if len(left) == 0 and len(right) == 0:
            return 0.0
            
        return np.max(np.abs(np.concatenate([left, right])))

    def generate_chirp(f1, f2):
        t = np.arange(N)
        phase = 2 * np.pi * f1 * t + np.pi * f2 * t**2
        sig = np.cos(phase)
        h = np.sign(sig)
        h[h == 0] = 1.0
        return h

    rng = np.random.RandomState(42)

    # 1. Initialization: Diverse population
    # Mix of chirps and random sequences
    population = []
    for i in range(POP_SIZE):
        if i < POP_SIZE // 2:
            f1 = rng.rand()
            f2 = rng.rand()
            h = generate_chirp(f1, f2)
        else:
            h = rng.choice([-1, 1], size=N).astype(np.float64)
        population.append(h)

    # Evaluate initial population
    scores = [get_psl(h) for h in population]
    
    # Track best overall
    best_idx = np.argmin(scores)
    best_h = population[best_idx].copy()
    best_score = scores[best_idx]

    # 2. Genetic Algorithm Loop
    # We use a "steady-state" GA approach where we generate a few offspring per generation
    # and replace the worst individuals if the offspring are better.
    # This preserves diversity better than generational replacement.
    
    # Parameters
    offspring_count = 5
    crossover_rate = 0.8
    mutation_rate = 0.05  # Probability of flipping a bit
    
    while time.time() < deadline:
        # Select parents
        # Use tournament selection or random selection from top half
        # To maintain diversity, we pick parents from the entire population but bias towards better ones
        # Simple approach: Randomly pick two parents from the population
        
        # Generate offspring
        offspring_list = []
        
        for _ in range(offspring_count):
            # Pick parents
            # Weighted selection: better parents have higher chance
            # Use softmax or simple top-k random
            # Let's pick from top 50% to keep quality high but not too elitist
            top_half_idx = np.argsort(scores)[:POP_SIZE // 2]
            p1_idx = top_half_idx[rng.randint(0, len(top_half_idx))]
            p2_idx = top_half_idx[rng.randint(0, len(top_half_idx))]
            
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            child = p1.copy()
            
            # Crossover: Uniform crossover
            if rng.rand() < crossover_rate:
                mask = rng.rand(N) < 0.5
                child[mask] = p2[mask]
            
            # Mutation: Bit flip
            if rng.rand() < mutation_rate:
                num_mutations = rng.randint(1, 5) # Flip 1 to 4 bits
                mut_indices = rng.choice(N, size=num_mutations, replace=False)
                child[mut_indices] *= -1.0
            
            # Evaluate offspring
            child_score = get_psl(child)
            offspring_list.append((child_score, child))
        
        # Selection and Replacement
        # Find the worst individual in the population
        worst_idx = np.argmax(scores)
        worst_score = scores[worst_idx]
        
        # Find the best offspring
        best_offspring_idx = np.argmin([s for s, _ in offspring_list])
        best_offspring_score, best_offspring = offspring_list[best_offspring_idx]
        
        # If the best offspring is better than the worst individual, replace it
        if best_offspring_score < worst_score:
            population[worst_idx] = best_offspring
            scores[worst_idx] = best_offspring_score
            
            # Update global best
            if best_offspring_score < best_score:
                best_score = best_offspring_score
                best_h = best_offspring.copy()
        
        # Additionally, if any offspring is better than the current global best, update it
        # (Already covered above if it replaces worst, but if worst is already very good, 
        # an offspring might be better than best_h but not better than worst? No, if it's better than best_h, 
        # it's likely better than many others. But let's just check against best_score directly)
        for s, c in offspring_list:
            if s < best_score:
                best_score = s
                best_h = c.copy()

    return best_h

if __name__ == "__main__":
    h = construct()
    print("Constructed h length:", len(h))
    print("First 10:", h[:10])
    # Calculate PSL for verification
    h_pad = np.zeros(2 * 400)
    h_pad[:400] = h
    H = np.fft.fft(h_pad)
    c = np.fft.ifft(H * np.conj(H)).real
    left = c[1:400]
    right = c[401:800]
    psl = np.max(np.abs(np.concatenate([left, right])))
    print("PSL:", psl)
