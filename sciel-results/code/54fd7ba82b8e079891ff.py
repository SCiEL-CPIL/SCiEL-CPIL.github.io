import time
import numpy as np
from itertools import combinations

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit

    def time_remaining():
        return time_limit - (time.time() - start_time)

    # Step 1: Construct the [17, 9, 6] QR code (256 words)
    # Quadratic residues modulo 17: {1, 2, 4, 8, 9, 13, 15, 16}
    # Include 0 to get dimension 9
    residues = [0, 1, 2, 4, 8, 9, 13, 15, 16]
    
    gen_row = [0] * N
    for r in residues:
        gen_row[r] = 1
    
    # Generate circulant matrix
    circ = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        for j in range(N):
            circ[i, j] = gen_row[(j - i) % N]
    
    G = circ[:9, :]
    
    # Pre-convert rows to integers
    g_ints = []
    for i in range(9):
        val = 0
        for j in range(N):
            if G[i, j] == 1:
                val |= (1 << j)
        g_ints.append(val)
    
    # Generate all 2^9 = 256 codewords
    base_words = []
    for mask in range(256):
        w = 0
        for i in range(9):
            if (mask >> i) & 1:
                w ^= g_ints[i]
        base_words.append(w)
    
    base_set = set(base_words)
    base_vecs = np.array([[(w >> i) & 1 for i in range(N)] for w in base_words], dtype=np.uint8)
    
    def is_compatible_with_base(candidate_vec):
        """Check if candidate has min distance from all base codewords."""
        dists = np.sum(base_vecs ^ candidate_vec, axis=1)
        return np.all(dists >= MIN_DIST)
    
    def to_bit_vector(x):
        return np.array([(x >> i) & 1 for i in range(N)], dtype=np.uint8)
    
    def from_bit_vector(vec):
        x = 0
        for i in range(N):
            if vec[i]:
                x |= (1 << i)
        return x
    
    # Step 2: Generate candidate words by weight
    # Focus on weights 6-11 (symmetric around 8.5 for N=17)
    # Candidates must be distance >= 6 from all base words
    
    candidates_by_weight = {}
    all_candidates = []
    candidate_indices = {}  # word -> index in all_candidates
    
    for weight in range(6, 12):
        # Generate all words of given weight
        candidates_by_weight[weight] = []
        
        # Use combinations to generate bit positions
        positions = list(range(N))
        for combo in combinations(positions, weight):
            w = 0
            for pos in combo:
                w |= (1 << pos)
            
            if w in base_set:
                continue
            
            vec = to_bit_vector(w)
            if is_compatible_with_base(vec):
                candidates_by_weight[weight].append(w)
                candidate_indices[w] = len(all_candidates)
                all_candidates.append(w)
    
    n_candidates = len(all_candidates)
    
    if time_remaining() < 5.0:
        return base_words
    
    # Step 3: Build compatibility graph among candidates
    # Edge between i and j if dist(candidates[i], candidates[j]) >= MIN_DIST
    # Use numpy for batch distance computation
    
    cand_vecs = np.array([to_bit_vector(w) for w in all_candidates], dtype=np.uint8)
    
    # Build adjacency as a list of sets for each node
    # For memory efficiency, use bitsets or sorted lists
    adjacency = [set() for _ in range(n_candidates)]
    
    # Batch compute distances
    # Process in chunks to avoid memory issues
    chunk_size = 5000
    for start in range(0, n_candidates, chunk_size):
        end = min(start + chunk_size, n_candidates)
        chunk_vecs = cand_vecs[start:end]
        
        # Compute distances from this chunk to all candidates
        # dist[i][j] = sum(chunk_vecs[i] ^ cand_vecs[j])
        for i in range(end - start):
            if time_remaining() < 3.0:
                break
            cv = chunk_vecs[i]
            dists = np.sum(cand_vecs ^ cv, axis=1)
            
            # Find candidates at distance >= MIN_DIST
            valid_indices = np.where(dists >= MIN_DIST)[0]
            
            for j in valid_indices:
                if j != start + i:
                    adjacency[start + i].add(j)
    
    if time_remaining() < 2.0:
        return base_words
    
    # Step 4: Greedy maximum clique on the compatibility graph
    # Use a greedy heuristic: repeatedly pick the vertex with highest degree
    # among remaining candidates, add to clique, remove it and its non-neighbors
    
    def greedy_max_clique(adjacency, n, time_budget):
        """Find a large clique using greedy degree-based heuristic."""
        if n == 0:
            return []
        
        # Work with a mutable copy of adjacency
        adj = [set(s) for s in adjacency]
        available = set(range(n))
        clique = []
        
        while available and time_budget() > 0.5:
            # Pick vertex with max degree in available subgraph
            best_v = -1
            best_deg = -1
            for v in available:
                deg = len(adj[v] & available)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                break
            
            clique.append(best_v)
            available.discard(best_v)
            
            # Remove non-neighbors of best_v from available
            neighbors = adj[best_v]
            to_remove = available - neighbors
            for v in to_remove:
                available.discard(v)
        
        return clique
    
    # Run greedy clique finding with multiple random orderings for diversity
    best_clique = []
    rng = np.random.default_rng(42)
    
    # Try several random perturbations of the greedy order
    for trial in range(20):
        if time_remaining() < 5.0:
            break
        
        # Shuffle vertex ordering to get different greedy paths
        shuffled_adj = [set() for _ in range(n_candidates)]
        for i in range(n_candidates):
            for j in adjacency[i]:
                shuffled_adj[i].add(j)
        
        clique = greedy_max_clique(shuffled_adj, n_candidates, time_remaining)
        
        if len(clique) > len(best_clique):
            best_clique = clique
    
    # Convert clique indices to words
    extra_words = [all_candidates[i] for i in best_clique]
    
    # Step 5: Combine base + clique
    all_words = base_words + extra_words
    all_set = set(all_words)
    
    # Step 6: Additional greedy addition from remaining candidates
    # Try to add more candidates not in the clique
    used_indices = set(best_clique)
    remaining_candidates = [i for i in range(n_candidates) if i not in used_indices]
    
    if remaining_candidates and time_remaining() > 5.0:
        # Build current code vector matrix
        current_vecs = np.array([to_bit_vector(w) for w in all_words], dtype=np.uint8)
        
        added = 0
        for idx in remaining_candidates:
            if time_remaining() < 2.0:
                break
            
            w = all_candidates[idx]
            vec = to_bit_vector(w)
            
            # Check against all current words
            dists = np.sum(current_vecs ^ vec, axis=1)
            if np.all(dists >= MIN_DIST):
                all_words.append(w)
                all_set.add(w)
                current_vecs = np.vstack([current_vecs, vec])
                added += 1
    
    # Step 7: Local search - try to swap out one word for two
    if time_remaining() > 10.0:
        n_current = len(all_words)
        current_vecs = np.array([to_bit_vector(w) for w in all_words], dtype=np.uint8)
        
        for _ in range(50):
            if time_remaining() < 5.0:
                break
            
            # Pick a random word to potentially remove
            idx = rng.integers(0, n_current)
            removed_word = all_words[idx]
            
            # Temporarily remove
            temp_vecs = np.delete(current_vecs, idx, axis=0)
            temp_words = all_words[:idx] + all_words[idx+1:]
            
            # Try to add two new words
            added_count = 0
            new_vecs = temp_vecs.copy()
            new_words = list(temp_words)
            
            # Try candidates from our pool
            for cand_idx in rng.choice(n_candidates, size=min(200, n_candidates), replace=False):
                if time_remaining() < 3.0:
                    break
                
                w = all_candidates[cand_idx]
                if w in set(new_words):
                    continue
                
                vec = to_bit_vector(w)
                dists = np.sum(new_vecs ^ vec, axis=1)
                if np.all(dists >= MIN_DIST):
                    new_words.append(w)
                    new_vecs = np.vstack([new_vecs, vec])
                    added_count += 1
                    if added_count >= 2:
                        break
            
            # Accept if we added at least 1 more word
            if added_count >= 1:
                all_words = new_words
                all_set = set(all_words)
                current_vecs = new_vecs
                n_current = len(all_words)
    
    # Remove duplicates
    unique_words = list(set(all_words))
    
    return unique_words
