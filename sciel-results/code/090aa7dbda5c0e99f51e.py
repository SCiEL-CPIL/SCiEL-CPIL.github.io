import time
import random
import numpy as np
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

def _compute_threshold_fast(trans0, trans1):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton.
    Uses bitmasks for fast subset transitions.
    Returns -1 if not synchronizing.
    """
    # Precompute transition masks for each state and letter
    t0 = [1 << trans0[i] for i in range(N)]
    t1 = [1 << trans1[i] for i in range(N)]
    
    # Helper to compute next subset
    def get_next(subset, letter):
        if letter == 0:
            res = 0
            s = subset
            while s:
                lsb = s & -s
                idx = (lsb.bit_length() - 1)
                res |= t0[idx]
                s ^= lsb
            return res
        else:
            res = 0
            s = subset
            while s:
                lsb = s & -s
                idx = (lsb.bit_length() - 1)
                res |= t1[idx]
                s ^= lsb
            return res

    # Precompute all transitions to make BFS O(1) per step
    next_subsets = [[0, 0] for _ in range(1 << N)]
    for s in range(1 << N):
        next_subsets[s][0] = get_next(s, 0)
        next_subsets[s][1] = get_next(s, 1)
        
    # BFS
    visited = [-1] * (1 << N)
    queue = deque()
    
    start = FULL_MASK
    visited[start] = 0
    queue.append(start)
    
    while queue:
        current = queue.popleft()
        dist = visited[current]
        
        # Check if singleton
        if current & (current - 1) == 0 and current != 0:
            return dist
            
        for letter in range(2):
            nxt = next_subsets[current][letter]
            if visited[nxt] == -1:
                visited[nxt] = dist + 1
                queue.append(nxt)
                
    return -1

def _generate_cerny_seed():
    """
    Generates a standard Cerny automaton which has threshold N-1.
    a: cycle (0->1->...->N-1->0)
    b: 0->0, i->i-1 for i>0
    """
    trans0 = [(i + 1) % N for i in range(N)]
    trans1 = [0] * N
    for i in range(1, N):
        trans1[i] = i - 1
    return trans0, trans1

def _diversity_score(t0, t1):
    """
    Heuristic score: Maximizes the average distance between image pairs.
    High diversity suggests slower synchronization.
    """
    # Count how many pairs of states map to distinct states under t0 and t1
    # If t0 is a permutation, it preserves all pairs.
    # If t1 is a permutation, it preserves all pairs.
    # We want to keep subsets large, so we want permutations where possible,
    # but we need at least one non-permutation to allow sync.
    
    # Simple heuristic: Count distinct images for each letter
    distinct0 = len(set(t0))
    distinct1 = len(set(t1))
    
    # We prefer one letter to be a permutation (12) and the other to be 
    # a "defective" permutation (11 or 10 distinct) to create the "long tail" structure.
    # If both are permutations, it won't sync.
    # If both are low rank, it syncs fast.
    
    # Score: Sum of distinctness, penalized if both are permutations (non-sync risk)
    # Actually, just returning distinctness sum is a good proxy for "potential length"
    # because high distinctness means less immediate collapse.
    
    # To ensure sync potential, we slightly penalize if both are full permutations
    if distinct0 == N and distinct1 == N:
        return 0 # Likely non-syncing or very hard to sync
    
    return distinct0 + distinct1

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s margin for safety
    
    rng = random.Random(42)
    
    best_0_global = None
    best_1_global = None
    best_score_global = 0
    
    # Start with the known best structure: Cerny Automaton
    # Then perform a "Beam Search" on the space of transitions.
    # We maintain a list of top candidates (beam) and expand them.
    
    # Initialize beam with Cerny seed and a few variations
    beam = []
    
    # Add Cerny seed
    c0, c1 = _generate_cerny_seed()
    beam.append((c0[:], c1[:], _compute_threshold_fast(c0, c1)))
    
    # Add some random near-permutation seeds to diversify
    for _ in range(20):
        t0 = list(range(N))
        rng.shuffle(t0)
        
        t1 = [(i + 1) % N for i in range(N)] # Start with cycle
        # Introduce 1-2 defects
        for _ in range(rng.randint(1, 2)):
            idx = rng.randint(0, N-1)
            t1[idx] = rng.randint(0, N-1)
            
        score = _compute_threshold_fast(t0, t1)
        if score > 0:
            beam.append((t0[:], t1[:], score))
            
    # Sort beam by score descending
    beam.sort(key=lambda x: x[2], reverse=True)
    
    # Keep top K candidates
    K = 50
    
    while time.time() < deadline:
        if not beam:
            break
            
        # Take the top candidate from the beam
        cur_0, cur_1, cur_score = beam[0]
        
        # If we found a new global best, update
        if cur_score > best_score_global:
            best_score_global = cur_score
            best_0_global = cur_0[:]
            best_1_global = cur_1[:]
            
        # Generate neighbors by mutating one transition
        # We prioritize mutations that maintain high diversity
        neighbors = []
        
        # Try mutating each state for letter 0
        for s in range(N):
            old_val = cur_0[s]
            for new_val in range(N):
                if new_val == old_val:
                    continue
                mut_0 = cur_0[:]
                mut_1 = cur_1[:]
                mut_0[s] = new_val
                
                # Quick diversity check before expensive BFS
                # If this mutation destroys the "structure" (e.g., makes both letters low rank), skip?
                # Or just compute score.
                
                new_score = _compute_threshold_fast(mut_0, mut_1)
                
                # We only keep valid (synchronizing) automata in the beam
                if new_score > 0:
                    neighbors.append((mut_0, mut_1, new_score))
                    
        # Try mutating each state for letter 1
        for s in range(N):
            old_val = cur_1[s]
            for new_val in range(N):
                if new_val == old_val:
                    continue
                mut_0 = cur_0[:]
                mut_1 = cur_1[:]
                mut_1[s] = new_val
                
                new_score = _compute_threshold_fast(mut_0, mut_1)
                
                if new_score > 0:
                    neighbors.append((mut_0, mut_1, new_score))

        # Merge neighbors into beam
        beam.extend(neighbors)
        
        # Deduplicate and keep top K
        # Using a set of tuples to deduplicate is expensive for lists, 
        # but N=12 is small. We can convert to tuple for hashing.
        seen = set()
        unique_beam = []
        for t0, t1, score in beam:
            key = (tuple(t0), tuple(t1))
            if key not in seen:
                seen.add(key)
                unique_beam.append((t0, t1, score))
        
        # Sort and keep top K
        unique_beam.sort(key=lambda x: x[2], reverse=True)
        beam = unique_beam[:K]
        
        # If the top of the beam hasn't improved in a while, we might be stuck.
        # But since we are exploring the neighborhood of the best, this is a local search.
        # To avoid infinite loops on the same set, we rely on the time limit.
        
    # Fallback
    if best_0_global is None:
        b0, b1 = _generate_cerny_seed()
        return [b0, b1]
        
    return [best_0_global, best_1_global]

# EVOLVE-BLOCK-END
