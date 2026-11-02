"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_fft(sig):
    """Compute autoconvolution of a signal using FFT."""
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    f = np.fft.rfft(sig, n_fft)
    ac = np.fft.irfft(f * np.conj(f), n_fft)[:2*N-1]
    return ac


def _exact_score(sig):
    """Compute C3' = max|autoconv| / (integral)^2."""
    integral = np.sum(sig)
    if abs(integral) < 1e-12:
        return float('inf')
    ac = _autoconv_fft(sig)
    max_abs = np.max(np.abs(ac))
    return max_abs / (integral * integral)


def _score_blocks(blocks):
    """Compute C3' for a block-constant signal given as list of (height, width)."""
    sig = np.zeros(N)
    pos = 0
    for h, w in blocks:
        sig[pos:pos+w] = h
        pos += w
    return _exact_score(sig)


def _blocks_to_sig(blocks):
    sig = np.zeros(N)
    pos = 0
    for h, w in blocks:
        sig[pos:pos+w] = h
        pos += w
    return sig


def _sig_to_blocks(sig):
    blocks = []
    i = 0
    while i < N:
        h = sig[i]
        j = i
        while j < N and sig[j] == h:
            j += 1
        blocks.append((h, j - i))
        i = j
    return blocks


def _merge_adjacent(blocks):
    """Merge adjacent blocks with the same sign."""
    if not blocks:
        return blocks
    merged = [blocks[0]]
    for h, w in blocks[1:]:
        if merged[-1][0] == h:
            merged[-1] = (h, merged[-1][1] + w)
        else:
            merged.append((h, w))
    return merged


def _local_search_blocks(blocks, deadline):
    """Local search on block representation: flip blocks, split blocks, re-merge."""
    best_blocks = list(blocks)
    best_score = _score_blocks(best_blocks)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try flipping each block
        for i in range(len(best_blocks)):
            if time.time() > deadline:
                return best_blocks, best_score
            new_blocks = list(best_blocks)
            h, w = new_blocks[i]
            new_blocks[i] = (-h, w)
            new_blocks = _merge_adjacent(new_blocks)
            s = _score_blocks(new_blocks)
            if s < best_score - 1e-12:
                best_score = s
                best_blocks = new_blocks
                improved = True
        
        # Try splitting a block and flipping the second part
        for i in range(len(best_blocks)):
            if time.time() > deadline:
                return best_blocks, best_score
            h, w = best_blocks[i]
            if w < 4:
                continue
            for frac in [0.25, 0.33, 0.5, 0.67, 0.75]:
                if time.time() > deadline:
                    return best_blocks, best_score
                w1 = max(1, int(w * frac))
                w2 = w - w1
                if w2 < 1:
                    continue
                new_blocks = list(best_blocks)
                new_blocks[i] = (h, w1)
                new_blocks.insert(i + 1, (-h, w2))
                new_blocks = _merge_adjacent(new_blocks)
                s = _score_blocks(new_blocks)
                if s < best_score - 1e-12:
                    best_score = s
                    best_blocks = new_blocks
                    improved = True
    
    return best_blocks, best_score


def _elementwise_local_search(sig, deadline):
    """Local search: flip individual elements."""
    best_score = _exact_score(sig)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            new_sig = sig.copy()
            new_sig[i] = -new_sig[i]
            s = _exact_score(new_sig)
            if s < best_score - 1e-12:
                best_score = s
                sig = new_sig
                improved = True
    
    return sig, best_score


def _genetic_annealing(deadline, pop_size=20):
    """Genetic algorithm with simulated annealing on discrete sign vectors."""
    rng = np.random.RandomState(12345)
    
    # Generate diverse initial population
    def random_structured_sig():
        """Generate a random structurally-constrained signal."""
        strategy = rng.randint(0, 6)
        sig = np.ones(N)
        if strategy == 0:
            # Two blocks
            split = rng.randint(N//4, 3*N//4)
            sig[split:] = -1.0
        elif strategy == 1:
            # Three blocks
            s1 = rng.randint(N//5, N//2)
            s2 = rng.randint(N//2, 4*N//5)
            sig[s1:s2] = -1.0
        elif strategy == 2:
            # Four blocks
            s1 = rng.randint(N//6, N//3)
            s2 = rng.randint(N//3, N//2)
            s3 = rng.randint(N//2, 2*N//3)
            sig[s1:s2] = -1.0
            sig[s2:s3] = 1.0
            sig[s3:] = -1.0
        elif strategy == 3:
            # Random with bias toward blocks
            n_blocks = rng.randint(2, 8)
            boundaries = np.sort(rng.choice(range(1, N), size=n_blocks-1, replace=False))
            for b in boundaries:
                sig[b:] *= -1.0
        elif strategy == 4:
            # Alternating blocks of random widths
            pos = 0
            sign = 1.0
            while pos < N:
                width = rng.randint(5, N//4 + 1)
                sig[pos:pos+width] = sign
                sign = -sign
                pos += width
        else:
            # Mostly random with some structure
            sig = rng.choice([-1.0, 1.0], size=N)
            # Smooth: flip runs of length > 3
            for i in range(N):
                if i+3 < N and sig[i] == sig[i+1] == sig[i+2] == sig[i+3]:
                    sig[i+1:i+3] *= -1.0
        return sig
    
    # Initialize population
    population = []
    scores = []
    
    # Add some structured initial candidates
    structured_inits = [
        lambda: np.concatenate([np.ones(N//2), -np.ones(N - N//2)]),
        lambda: np.concatenate([np.ones(N//3), -np.ones(N//3), np.ones(N - 2*N//3)]),
        lambda: np.concatenate([np.ones(N//4), -np.ones(N//4), np.ones(N//4), -np.ones(N - 3*N//4)]),
        random_structured_sig,
        random_structured_sig,
        random_structured_sig,
    ]
    
    for init_func in structured_inits:
        s = init_func()
        sc = _exact_score(s)
        population.append(s)
        scores.append(sc)
    
    # Fill rest with random
    while len(population) < pop_size:
        s = random_structured_sig()
        sc = _exact_score(s)
        population.append(s)
        scores.append(sc)
    
    best_idx = np.argmin(scores)
    best_sig = population[best_idx].copy()
    best_score = scores[best_idx]
    
    # Genetic algorithm with annealing
    T_start = max(best_score * 0.1, 0.01)
    T_end = T_start * 0.01
    cooling_rate = 0.95
    
    T = T_start
    no_improve_count = 0
    max_no_improve = 30
    
    while time.time() < deadline:
        # Check if we should stop
        if no_improve_count > max_no_improve and T < T_end * 10:
            break
        
        new_population = []
        new_scores = []
        
        # Keep best
        new_population.append(best_sig.copy())
        new_scores.append(best_score)
        
        # Elitism: keep top 2
        sorted_indices = np.argsort(scores)
        for idx in sorted_indices[:2]:
            if idx == best_idx:
                continue
            if len(new_population) >= 3:
                break
            new_population.append(population[idx].copy())
            new_scores.append(scores[idx])
        
        # Generate offspring until population is full
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
            
            op = rng.random()
            
            if op < 0.4:
                # Crossover: take first part from one parent, second from another
                p1_idx = rng.randint(0, len(population))
                p2_idx = rng.randint(0, len(population))
                if p1_idx == p2_idx:
                    p2_idx = (p2_idx + 1) % len(population)
                cross_point = rng.randint(N//4, 3*N//4)
                child = np.concatenate([population[p1_idx][:cross_point], 
                                        population[p2_idx][cross_point:]])
            elif op < 0.7:
                # Mutation: flip a random segment
                parent_idx = rng.randint(0, len(population))
                child = population[parent_idx].copy()
                seg_start = rng.randint(0, N-1)
                seg_len = rng.randint(1, N//5 + 1)
                child[seg_start:seg_start+seg_len] *= -1.0
            elif op < 0.85:
                # Point mutations: flip several random positions
                parent_idx = rng.randint(0, len(population))
                child = population[parent_idx].copy()
                n_mut = rng.randint(1, 5)
                mut_positions = rng.choice(N, size=n_mut, replace=False)
                child[mut_positions] *= -1.0
            else:
                # Fresh random
                child = random_structured_sig()
            
            child_score = _exact_score(child)
            
            # Simulated annealing acceptance
            if len(new_population) == 0 or child_score < min(new_scores):
                new_population.append(child)
                new_scores.append(child_score)
            else:
                # Accept with probability exp(-delta/T)
                delta = child_score - min(new_scores)
                if T > 0 and rng.random() < np.exp(-delta / T):
                    new_population.append(child)
                    new_scores.append(child_score)
                else:
                    # Replace worst
                    worst_idx = np.argmax(new_scores)
                    new_population[worst_idx] = child
                    new_scores[worst_idx] = child_score
            
            # Update best
            if child_score < best_score:
                best_score = child_score
                best_sig = child.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        population = new_population
        scores = new_scores
        T *= cooling_rate
        
        # Periodic local refinement of best
        if no_improve_count % 10 == 0 and time.time() < deadline - 5:
            refined, ref_score = _elementwise_local_search(best_sig, min(time.time() + 2, deadline))
            if ref_score < best_score:
                best_score = ref_score
                best_sig = refined.copy()
    
    return best_sig, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Phase 1: Genetic algorithm with simulated annealing
    best_sig, best_score = _genetic_annealing(deadline, pop_size=20)
    
    if best_sig is None:
        # Fallback: simple two-block
        best_sig = np.zeros(N)
        best_sig[:N//2] = 1.0
        best_sig[N//2:] = -1.0
        best_score = _exact_score(best_sig)
    
    if time.time() > deadline:
        return best_sig
    
    # Phase 2: Block-based local search (structural refinement)
    blocks = _sig_to_blocks(best_sig)
    blocks, block_score = _local_search_blocks(blocks, deadline)
    if block_score < best_score:
        best_score = block_score
        best_sig = _blocks_to_sig(blocks)
    
    if time.time() > deadline:
        return best_sig
    
    # Phase 3: Element-wise local search (fine-grained refinement)
    refined_sig, refined_score = _elementwise_local_search(best_sig, deadline)
    if refined_score < best_score:
        best_sig = refined_sig
        best_score = refined_score
    
    # Phase 4: Final block-based pass on the refined signal
    if time.time() < deadline - 5.0:
        blocks = _sig_to_blocks(best_sig)
        blocks, block_score = _local_search_blocks(blocks, deadline)
        if block_score < best_score:
            best_score = block_score
            best_sig = _blocks_to_sig(blocks)
    
    return best_sig
# EVOLVE-BLOCK-END
