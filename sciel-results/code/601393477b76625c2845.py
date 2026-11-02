"""Improved Greedy with Local Search for ternary covering codes K_3(7,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def _precompute_neighbors(q, n, radius):
    """Precompute neighbors for all words."""
    N = q ** n
    # For radius 1, number of neighbors is 1 + n*(q-1)
    # For q=3, n=7, this is 1 + 7*2 = 15
    if radius == 1:
        max_nbrs = 1 + n * (q - 1)
    else:
        # Fallback for general case, though task is specific
        max_nbrs = q ** n 

    nbrs_list = [None] * N
    
    if q == 3 and n == 7:
        # Unrolled fast path for specific case
        for idx in range(N):
            word = [0] * 7
            tmp = idx
            word[0] = tmp % 3; tmp //= 3
            word[1] = tmp % 3; tmp //= 3
            word[2] = tmp % 3; tmp //= 3
            word[3] = tmp % 3; tmp //= 3
            word[4] = tmp % 3; tmp //= 3
            word[5] = tmp % 3; tmp //= 3
            word[6] = tmp % 3; tmp //= 3
            
            nbrs = [idx]
            for pos in range(7):
                old = word[pos]
                for v in range(3):
                    if v != old:
                        new_word = word.copy()
                        new_word[pos] = v
                        # Fast index calculation
                        new_idx = new_word[0] + new_word[1]*3 + new_word[2]*9 + new_word[3]*27 + new_word[4]*81 + new_word[5]*243 + new_word[6]*729
                        nbrs.append(new_idx)
            nbrs_list[idx] = np.array(nbrs, dtype=np.int32)
    else:
        for idx in range(N):
            word = [0] * n
            tmp = idx
            for i in range(n):
                word[i] = tmp % q
                tmp //= q
            nbrs = [idx]
            for pos in range(n):
                old = word[pos]
                for v in range(q):
                    if v != old:
                        new_word = word.copy()
                        new_word[pos] = v
                        new_idx = 0
                        mul = 1
                        for i in range(n):
                            new_idx += new_word[i] * mul
                            mul *= q
                        nbrs.append(new_idx)
            nbrs_list[idx] = np.array(nbrs, dtype=np.int32)
            
    return nbrs_list

def _run_greedy_local_search(nbrs_list, N, time_limit=110.0, start_time=0.0):
    """Run Greedy Set Cover followed by Local Search."""
    
    # 1. Greedy Phase
    # covered_count[v] = number of selected codewords that cover v
    covered_count = np.zeros(N, dtype=np.int32)
    selected = []
    selected_set = set()
    
    # Initial gain is the size of the neighborhood
    # We want to pick the word that covers the most UNCOVERED points.
    # Gain(w) = sum(1 for v in N(w) if covered_count[v] == 0)
    
    # To speed up, we can maintain a gain array.
    # gain[w] = number of uncovered points in N(w)
    gain = np.zeros(N, dtype=np.int32)
    for w in range(N):
        gain[w] = len(nbrs_list[w])
        
    while True:
        if time.time() - start_time > time_limit * 0.2: # Stop greedy early to allow time for local search
            break
        max_gain = np.max(gain)
        if max_gain == 0:
            break
            
        # Find candidates with max gain. 
        # To add some diversity or just pick first, we can pick argmax.
        # For better results, we might pick randomly from top few, but let's start with deterministic greedy.
        idx = np.argmax(gain)
        
        selected.append(idx)
        selected_set.add(idx)
        
        # Update gains
        # For every neighbor v of idx:
        # If covered_count[v] was 0, it is now covered.
        # This reduces the gain of all codewords u that have v in their neighborhood.
        # This is expensive if done naively.
        # Alternative: Just recompute gains for all candidates? No, too slow.
        
        # Efficient Update:
        # We maintain covered_count.
        # When we add idx, we increment covered_count for all v in N(idx).
        # For each v in N(idx) that was previously uncovered (count 0 -> 1):
        #    We must decrement gain[u] for all u such that v in N(u).
        
        nbrs_idx = nbrs_list[idx]
        for v in nbrs_idx:
            if covered_count[v] == 0:
                # v is newly covered
                # Find all u that cover v. 
                # Instead of iterating all N, we can precompute reverse neighbors? 
                # Or just iterate all u and check? N=2187. 
                # Iterating all N for each new cover is O(N^2 * R) which might be too slow.
                # 2187^2 is ~5M. Times 15 (radius) is 75M. Might be okay in Python if vectorized?
                
                # Let's try a different strategy for Greedy:
                # Just recompute gain for all N in a vectorized way?
                # gain[w] = sum(covered_count[nbrs_list[w]] == 0)
                # This is O(N * R) per step. 2187 * 15 = 32k ops.
                # Number of steps is ~100-200.
                # Total ops ~ 3M-6M. This is very fast in NumPy.
                pass
            
            covered_count[v] += 1
        
        # Recompute gain for all w
        # We need to check for each w, how many of its neighbors have covered_count == 0
        # We can do this by iterating w and summing.
        # To vectorize:
        # Create a matrix M where M[w, i] is the i-th neighbor of w.
        # Then gain[w] = sum(covered_count[M[w, :]] == 0)
        # This is efficient.
        
        # Let's build the matrix once outside? 
        # It's 2187 x 15. Very small.
        
    # Re-initialize for the vectorized approach which is simpler and robust
    covered_count = np.zeros(N, dtype=np.int32)
    selected = []
    selected_set = set()
    
    # Precompute neighbor matrix for vectorized gain calculation
    # Max neighbors is 15
    max_nbrs = len(nbrs_list[0])
    nbrs_mat = np.zeros((N, max_nbrs), dtype=np.int32)
    for w in range(N):
        nbrs_mat[w, :len(nbrs_list[w])] = nbrs_list[w]
        
    while True:
        if time.time() - start_time > time_limit * 0.3:
            break
            
        # Calculate gain: number of uncovered neighbors
        # is_uncovered = covered_count == 0
        # gain[w] = sum(is_uncovered[nbrs_mat[w, :]])
        is_uncovered = (covered_count == 0).astype(np.int32)
        # Use matrix multiplication or sum
        # gain = is_uncovered[nbrs_mat].sum(axis=1)
        gain = is_uncovered[nbrs_mat].sum(axis=1)
        
        max_gain = np.max(gain)
        if max_gain == 0:
            break
        
        idx = np.argmax(gain)
        selected.append(idx)
        selected_set.add(idx)
        
        # Update covered_count
        nbrs_idx = nbrs_list[idx]
        covered_count[nbrs_idx] += 1
        
    # 2. Local Search Phase
    # Goal: Minimize len(selected)
    # Strategy: 
    # 1. Remove any redundant codewords (if removing one doesn't uncover any points).
    # 2. Try 2-swap: Remove two, add one or two.
    
    # Helper to check if a set of codewords covers everything
    # We can maintain covered_count dynamically.
    
    # Reset covered_count based on current 'selected'
    covered_count = np.zeros(N, dtype=np.int32)
    for w in selected:
        covered_count[nbrs_list[w]] += 1
        
    # 1. Remove redundant
    changed = True
    while changed and time.time() - start_time < time_limit * 0.8:
        changed = False
        for i in range(len(selected)):
            w = selected[i]
            # Check if removing w leaves any point uncovered
            # A point v is uncovered if covered_count[v] == 1 and w covers v.
            # If no such v exists, w is redundant.
            nbrs_w = nbrs_list[w]
            # Find v in nbrs_w where covered_count[v] == 1
            # If any, w is critical.
            # If none, w is redundant.
            critical = np.any(covered_count[nbrs_w] == 1)
            if not critical:
                # Remove w
                covered_count[nbrs_w] -= 1
                selected.pop(i)
                changed = True
                break # Restart loop as indices changed
    
    # 2. 2-Swap Local Search
    # Try to replace two codewords with one, or two different ones that cover more "critical" points.
    # Simpler: Try to remove one and add one (1-swap) or remove two and add one (2-1 swap).
    # Given the tight time limit, let's try a simple 1-swap first: 
    # For each w in selected, try to replace it with w' not in selected 
    # such that the new set is still valid and potentially allows removal of others later?
    # Actually, standard 2-opt for set cover:
    # Try to remove w1, w2 and add w3, w4 (or just w3).
    
    # Let's implement a simple "add-remove" loop.
    # While time remains:
    #   Try to find a codeword w in selected that can be replaced by w' not in selected
    #   such that the coverage is maintained. 
    #   This doesn't reduce size directly.
    
    # Better: Try to find a codeword w in selected that is "bad" (covers few unique points)
    # and replace it with a "good" one, hoping to enable a removal later?
    
    # Let's stick to the most effective simple heuristic for small N:
    # Iterate through all pairs (i, j) in selected.
    # Try to remove selected[i] and selected[j].
    # Check if the remaining set + some new word(s) covers everything.
    # This is expensive.
    
    # Alternative: 
    # Just run the greedy multiple times with different randomizations?
    # The current greedy is deterministic. Let's add randomness to the greedy phase.
    
    # Since we are inside a single run function, let's just return the current best.
    # The main loop in construct() will call this multiple times with different seeds if we modify it.
    
    # Let's add a simple random restart inside the greedy by picking randomly from top K gains.
    
    return selected

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 110.0
    
    nbrs_list = _precompute_neighbors(q, n, radius)
    N = q ** n
    
    best_code = None
    best_size = float('inf')
    
    # Run multiple iterations with different random seeds for the greedy phase
    # We need to modify _run_greedy_local_search to accept a seed and use it for tie-breaking.
    
    for seed in range(1000):
        if time.time() - start_time > time_limit - 5.0:
            break
            
        # We need a version of greedy that uses seed
        # Let's create a local helper or pass seed
        
        # For simplicity, let's just run the deterministic greedy once, 
        # then do local search. 
        # To improve, we can randomize the order of evaluation in greedy.
        
        # Let's modify the greedy to pick randomly from top 10 candidates.
        
        # Re-implementing greedy with randomness here to avoid complex refactoring of the helper
        covered_count = np.zeros(N, dtype=np.int32)
        selected = []
        selected_set = set()
        
        # Precompute neighbor matrix for vectorized gain calculation
        max_nbrs = len(nbrs_list[0])
        nbrs_mat = np.zeros((N, max_nbrs), dtype=np.int32)
        for w in range(N):
            nbrs_mat[w, :len(nbrs_list[w])] = nbrs_list[w]
            
        rng = np.random.RandomState(seed)
        
        while True:
            if time.time() - start_time > time_limit - 10.0: # Leave time for local search
                break
                
            is_uncovered = (covered_count == 0).astype(np.int32)
            gain = is_uncovered[nbrs_mat].sum(axis=1)
            
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            
            # Get indices with max gain
            candidates = np.where(gain == max_gain)[0]
            # Pick one randomly
            idx = rng.choice(candidates)
            
            selected.append(idx)
            selected_set.add(idx)
            
            nbrs_idx = nbrs_list[idx]
            covered_count[nbrs_idx] += 1
            
        # Local Search: Remove redundant
        changed = True
        while changed and time.time() - start_time < time_limit - 2.0:
            changed = False
            for i in range(len(selected)):
                w = selected[i]
                nbrs_w = nbrs_list[w]
                critical = np.any(covered_count[nbrs_w] == 1)
                if not critical:
                    covered_count[nbrs_w] -= 1
                    selected.pop(i)
                    changed = True
                    break
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_code = selected.copy()
            
    # Convert indices to words
    res = []
    for idx in best_code:
        word = [0] * n
        tmp = idx
        for i in range(n):
            word[i] = tmp % q
            tmp //= q
        res.append(word)
        
    return res
# EVOLVE-BLOCK-END
