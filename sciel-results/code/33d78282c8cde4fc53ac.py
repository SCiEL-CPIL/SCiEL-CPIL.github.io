"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius `radius`.
    Uses a greedy set cover approach followed by local search (deletion/swap) to minimize size.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Handle trivial cases or specific known structures
    if q == 3 and n == 7 and radius == 1:
        # Known optimal is 186. We try to find a small code.
        # Start with a reasonable initial solution.
        # The direct product construction gives 9 * 3^3 = 243.
        # We can use a better initial seed or just run the optimizer on the 243 set.
        
        # Generate all 3^7 = 2187 words
        all_words = list(itertools.product(range(q), repeat=n))
        N = len(all_words)
        
        # Precompute neighbors (distance <= radius) for each word in the space
        # Since n=7, q=3, radius=1, each word has 1 + 7*2 = 15 neighbors.
        # We can represent the set of covered words as a bitset (integer or list of bools)
        # 2187 bits is small enough to fit in a few integers or a Python int.
        
        # Map word to index
        word_to_idx = {}
        for i, w in enumerate(all_words):
            word_to_idx[w] = i
            
        # Precompute neighbors for each index
        neighbors = [set() for _ in range(N)]
        for i, w in enumerate(all_words):
            neighbors[i].add(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_w = list(w)
                    new_w[pos] = val
                    idx = word_to_idx[tuple(new_w)]
                    neighbors[i].add(idx)
        
        # Convert neighbors to lists for faster iteration
        neighbors_list = [list(s) for s in neighbors]
        
        # Initial Solution: Greedy Set Cover
        # Start with a random or structured subset.
        # Let's start with the 243 word construction and then try to remove.
        # Or better, start with a random large set and reduce?
        # Greedy Set Cover is usually a good upper bound.
        
        # Let's try starting with the 243 words from the seed program, then delete.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_codewords = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        codeword_indices = [word_to_idx[c] for c in initial_codewords]
        
        # Optimize: Local Search
        # Objective: Minimize number of codewords.
        # Constraint: Union of neighbors(codewords) == All Words.
        
        # We can use a bitset for covered words.
        # Python integers can act as arbitrary precision bitsets.
        covered_mask = 0
        for idx in codeword_indices:
            for nb in neighbors[idx]:
                covered_mask |= (1 << nb)
        
        full_mask = (1 << N) - 1
        
        # Check if covered
        if covered_mask != full_mask:
             # This shouldn't happen with the seed construction, but just in case
             pass

        # Local Search Loop
        # Try to remove a codeword if it doesn't uncover any word.
        # Try to swap a codeword with a non-codeword if it reduces size (impossible for swap) 
        # or maintains size but allows further removal? 
        # Actually, standard reduction:
        # 1. Try to remove any codeword that is "redundant".
        # 2. If no redundant, try to "move" a codeword to a position that allows removal of another?
        #    This is complex. 
        #    Alternative: Start with a larger random set and greedily remove.
        #    Or: Start with the 243 set, and try to replace pairs?
        
        # Let's implement a simple "remove if redundant" loop.
        # To break out of local minima, we can use random restarts or simulated annealing on the set size.
        # Given the time limit, let's try multiple restarts with random initial sets or perturbations.
        
        best_size = len(codeword_indices)
        best_set = set(codeword_indices)
        
        # Strategy 1: Greedy Removal from the 243 set
        current_set = list(codeword_indices)
        # Shuffle to remove in random order to avoid bias
        random.shuffle(current_set)
        
        # Rebuild mask
        covered_mask = 0
        for idx in current_set:
            for nb in neighbors[idx]:
                covered_mask |= (1 << nb)
        
        # We need to know for each word, how many codewords cover it?
        # Or simpler: for each candidate codeword to remove, check if all its neighbors are covered by others.
        # Efficient check: 
        # For each word w in neighbors[idx], is there another codeword c in current_set \ {idx} such that w in neighbors[c]?
        # This is O(|neighbors[idx]| * |current_set|) per removal attempt. 
        # |neighbors| = 15. |current_set| ~ 200. 15*200 = 3000 ops per removal.
        # 200 removals * 3000 = 600k ops. Very fast.
        
        # Let's optimize the check:
        # Maintain a count array: cover_count[w] = number of codewords in current_set that cover w.
        # If min(cover_count[w] for w in neighbors[idx]) > 1, then idx can be removed.
        
        cover_count = [0] * N
        for idx in current_set:
            for nb in neighbors[idx]:
                cover_count[nb] += 1
                
        removed_any = True
        while removed_any and time.time() < deadline:
            removed_any = False
            # Iterate in random order
            random.shuffle(current_set)
            for idx in current_set:
                if time.time() > deadline: break
                # Check if redundant
                redundant = True
                for nb in neighbors[idx]:
                    if cover_count[nb] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove idx
                    for nb in neighbors[idx]:
                        cover_count[nb] -= 1
                    current_set.remove(idx)
                    removed_any = True
                    best_size = len(current_set)
                    best_set = set(current_set)
            
        # If we have time, try random restarts with different initial sets
        # Or perturb the current best set.
        
        # Let's try a few random restarts from scratch with greedy set cover?
        # Greedy set cover:
        # 1. Start with empty set.
        # 2. Add word that covers most uncovered words.
        # This is O(N * |Uncovered|) per step.
        
        # Let's try one greedy set cover from scratch if time permits.
        if time.time() < deadline - 10:
            uncovered = set(range(N))
            greedy_set = []
            # Precompute neighbor lists as lists for faster iteration
            # We already have neighbors as sets.
            
            # To speed up greedy, we can use a heap or just scan.
            # Scanning 2187 words * 15 neighbors is ~30k ops.
            # 200 steps * 30k = 6M ops. Feasible.
            
            while uncovered and time.time() < deadline - 5:
                best_idx = -1
                best_count = 0
                for idx in range(N):
                    if idx in greedy_set: continue
                    # Count how many uncovered words idx covers
                    cnt = 0
                    for nb in neighbors[idx]:
                        if nb in uncovered:
                            cnt += 1
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = idx
                    if cnt == len(uncovered): # Perfect cover found
                        break
                if best_idx == -1: break
                greedy_set.append(best_idx)
                # Update uncovered
                for nb in neighbors[best_idx]:
                    uncovered.discard(nb)
                
            if len(greedy_set) < best_size:
                best_size = len(greedy_set)
                best_set = set(greedy_set)
                # Try to reduce this greedy set as well
                current_set = list(greedy_set)
                cover_count = [0] * N
                for idx in current_set:
                    for nb in neighbors[idx]:
                        cover_count[nb] += 1
                removed_any = True
                while removed_any and time.time() < deadline - 2:
                    removed_any = False
                    random.shuffle(current_set)
                    for idx in current_set:
                        if time.time() > deadline - 2: break
                        redundant = True
                        for nb in neighbors[idx]:
                            if cover_count[nb] <= 1:
                                redundant = False
                                break
                        if redundant:
                            for nb in neighbors[idx]:
                                cover_count[nb] -= 1
                            current_set.remove(idx)
                            removed_any = True
                            if len(current_set) < best_size:
                                best_size = len(current_set)
                                best_set = set(current_set)
        
        # Return the best set found
        result = [list(all_words[idx]) for idx in best_set]
        return result

    # Generic fallback for other parameters (not used in this specific task instance but good for robustness)
    # Simple greedy set cover
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    neighbors = []
    for w in all_words:
        nbs = {word_to_idx[w]}
        for pos in range(n):
            for val in range(q):
                if val == w[pos]: continue
                nw = list(w)
                nw[pos] = val
                nbs.add(word_to_idx[tuple(nw)])
        neighbors.append(nbs)
    
    uncovered = set(range(len(all_words)))
    code_set = []
    while uncovered and time.time() < deadline:
        best_idx = -1
        best_count = 0
        for idx in range(len(all_words)):
            if idx in code_set: continue
            cnt = sum(1 for nb in neighbors[idx] if nb in uncovered)
            if cnt > best_count:
                best_count = cnt
                best_idx = idx
        if best_idx == -1: break
        code_set.append(best_idx)
        for nb in neighbors[best_idx]:
            uncovered.discard(nb)
    
    return [list(all_words[idx]) for idx in code_set]
# EVOLVE-BLOCK-END
