import numpy as np
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    N = q ** n
    # Precompute neighbors for all words
    # neighbors[i] is a list of words within distance <= radius from i
    # For radius 1, size is 1 + n*(q-1)
    neighbors = []
    for w in range(N):
        neigh = [w]
        for i in range(n):
            power = q ** i
            digit = (w // power) % q
            for d in range(q):
                if d != digit:
                    new_w = w + (d - digit) * power
                    neigh.append(new_w)
        neighbors.append(neigh)
    
    # Helper to convert int to list of digits
    def to_digits(w):
        return [int((w // (q**i)) % q) for i in range(n)]

    best_code = None
    best_len = float('inf')
    
    # Run multiple trials with randomization
    num_trials = 10
    for t in range(num_trials):
        if time.time() > deadline - 5.0:
            break
            
        # 1. Greedy Initialization
        # uncovered: 1 if not covered, 0 if covered
        uncovered = np.ones(N, dtype=np.int8)
        # coverage_count[i]: number of uncovered words covered by i
        # Initially, all are covered by 0 codewords, so uncovered count is len(neighbors[i])
        # But we want to pick the one covering most UNCOVERED words.
        # So coverage_count[i] = sum(uncovered[w] for w in neighbors[i])
        
        # Initial coverage count
        cov_count = np.array([len(neigh) for neigh in neighbors], dtype=np.int32)
        
        code = []
        remaining = N
        
        while remaining > 0:
            if time.time() > deadline - 5.0:
                break
            
            # Add some randomness to tie-breaking or selection
            # Find max cov_count
            # To add randomness, we can pick from top K
            # For simplicity, just pick max, but shuffle neighbors in precomputation? 
            # Or just run multiple greedy passes.
            
            max_idx = np.argmax(cov_count)
            
            # Add to code
            code.append(max_idx)
            
            # Update uncovered and cov_count
            for u in neighbors[max_idx]:
                if uncovered[u]:
                    uncovered[u] = 0
                    remaining -= 1
                    # Decrement cov_count for all v that cover u
                    for v in neighbors[u]:
                        cov_count[v] -= 1
                        
        if remaining > 0:
            # Timeout during greedy, try to finish or skip
            # If we break early, code is invalid.
            # We must ensure validity.
            # Let's just force finish if close, or discard this trial.
            # Greedy on 6561 words is fast.
            pass

        # 2. Reduction Phase
        # Compute cover counts for the current code
        counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for u in neighbors[c]:
                counts[u] += 1
        
        # Check validity (min count >= 1)
        # If invalid, we need to fix it. But greedy ensures validity.
        
        # Remove redundant codewords
        changed = True
        while changed:
            changed = False
            # Iterate through code
            # We need to check if a codeword is essential.
            # c is essential if there exists w in B(c) such that counts[w] == 1.
            
            to_remove = []
            for c in code:
                is_essential = False
                for u in neighbors[c]:
                    if counts[u] == 1:
                        is_essential = True
                        break
                if not is_essential:
                    to_remove.append(c)
            
            if to_remove:
                for c in to_remove:
                    code.remove(c)
                    for u in neighbors[c]:
                        counts[u] -= 1
                changed = True

        if len(code) < best_len:
            best_len = len(code)
            best_code = code[:]

        # 3. Swap Phase (2-for-1)
        # Try to replace pairs with single codewords
        # This is O(|code|^2 * N) roughly if done naively.
        # |code| ~ 500. 500^2 = 250k. N=6561.
        # 250k * 6561 is too big.
        # We need to be smarter.
        
        # Only try swaps if time allows.
        if time.time() < deadline - 10.0:
             # Recompute counts if we didn't just do it
             # counts is already up to date from reduction.
             
             # Identify "weak" codewords?
             # Just iterate through pairs.
             # To speed up, we can precompute for each word w, the list of codewords covering it?
             # No, we have counts.
             
             # Strategy:
             # For each pair (i, j), check if B(i) U B(j) can be covered by Rest + {k}
             # This is equivalent to:
             # Let U be the set of words w such that counts[w] - I(w in B(i)) - I(w in B(j)) == 0.
             # If U is empty, remove i and j.
             # Else, check if there exists k not in code such that U is subset of B(k).
             
             # We will limit the number of swap iterations.
             swap_improved = True
             while swap_improved:
                 swap_improved = False
                 if time.time() > deadline - 5.0:
                     break
                 
                 # Shuffle code to avoid bias?
                 # We need to check pairs.
                 # To make it faster, we can just check a subset of pairs or use heuristics.
                 # Given the small size, let's try all pairs but break early if no improvement found in a full pass.
                 
                 n_code = len(code)
                 # Create a set for fast lookup of code indices
                 code_set = set(code)
                 
                 # Iterate through pairs
                 # We need to be careful with modifying code while iterating.
                 # We'll collect moves to apply later.
                 moves = []
                 
                 for a_idx in range(n_code):
                     if time.time() > deadline - 5.0:
                         break
                     c_i = code[a_idx]
                     for b_idx in range(a_idx + 1, n_code):
                         c_j = code[b_idx]
                         
                         # Calculate U (words that would become uncovered)
                         # We need to know counts for words in B(c_i) U B(c_j)
                         # Let's just iterate through the union.
                         # To avoid duplicates, use a set or just check both.
                         
                         uncovered_by_pair = set()
                         
                         # Check words in B(c_i)
                         for u in neighbors[c_i]:
                             # New count if we remove c_i and c_j
                             # counts[u] - (1 if u in B(c_i) else 0) - (1 if u in B(c_j) else 0)
                             # Since u is in B(c_i), first term is 1.
                             new_count = counts[u] - 1
                             if u in neighbors[c_j]:
                                 new_count -= 1
                                 
                             if new_count < 1:
                                 uncovered_by_pair.add(u)
                                 
                         # Check words in B(c_j) that are not in B(c_i)
                         # We already checked all in B(c_i).
                         # We need to check u in B(c_j) such that u not in B(c_i).
                         # To do this fast, we can just check all in B(c_j) and skip if in B(c_i)?
                         # Or just check all in B(c_j) and update uncovered_by_pair.
                         # If u was in B(c_i), we already processed it.
                         # If u is in both, new_count = counts[u] - 2.
                         # If u is only in B(c_j), new_count = counts[u] - 1.
                         
                         # Let's just iterate B(c_j) and handle logic.
                         for u in neighbors[c_j]:
                             if u in neighbors[c_i]:
                                 # Already handled in first loop?
                                 # Yes, if we added to set.
                                 # But we need to ensure we don't double count or miss.
                                 # The set `uncovered_by_pair` handles uniqueness.
                                 # The logic `new_count = counts[u] - 2` is correct for intersection.
                                 # In the first loop, we did `new_count = counts[u] - 1` then `if u in B(c_j): new_count -= 1`.
                                 # So it is handled.
                                 pass
                             else:
                                 new_count = counts[u] - 1
                                 if new_count < 1:
                                     uncovered_by_pair.add(u)
                                     
                         if not uncovered_by_pair:
                             # We can remove both
                             moves.append(('remove', c_i, c_j))
                             swap_improved = True
                         else:
                             # Try to find a replacement k
                             # U = uncovered_by_pair
                             # Find k not in code such that U subset of B(k)
                             
                             # Optimization: If |U| is large, unlikely to find k.
                             # Also, only check k that are "good" candidates?
                             # For now, check all k not in code.
                             # To speed up, we can precompute for each word w, the list of k covering w?
                             # Let's just iterate k.
                             
                             found_k = None
                             for k in range(N):
                                 if k in code_set:
                                     continue
                                 # Check if U subset of B(k)
                                 # i.e., all u in U are in neighbors[k]
                                 # We can break early if one u is not in neighbors[k]
                                 valid = True
                                 for u in uncovered_by_pair:
                                     if u not in neighbors[k]:
                                         valid = False
                                         break
                                 if valid:
                                     found_k = k
                                     break
                             
                             if found_k is not None:
                                 moves.append(('swap', c_i, c_j, found_k))
                                 swap_improved = True
                 
                 # Apply moves
                 # Note: Applying moves changes counts and code.
                 # We must be careful.
                 # If we do multiple moves, they might interfere.
                 # Safest: Apply one move at a time and restart the pair search?
                 # Or apply all non-conflicting moves.
                 # Given the complexity, let's just apply the first found move and restart.
                 # This is slower but safer.
                 
                 if moves:
                     # Apply first move
                     move = moves[0]
                     if move[0] == 'remove':
                         _, ci, cj = move
                         code.remove(ci)
                         code.remove(cj)
                         for u in neighbors[ci]:
                             counts[u] -= 1
                         for u in neighbors[cj]:
                             counts[u] -= 1
                         code_set.discard(ci)
                         code_set.discard(cj)
                     else:
                         _, ci, cj, ck = move
                         code.remove(ci)
                         code.remove(cj)
                         code.append(ck)
                         code_set.discard(ci)
                         code_set.discard(cj)
                         code_set.add(ck)
                         
                         for u in neighbors[ci]:
                             counts[u] -= 1
                         for u in neighbors[cj]:
                             counts[u] -= 1
                         for u in neighbors[ck]:
                             counts[u] += 1
                         
                     # After a move, we should run reduction again
                     # And restart the swap loop
                     # So we break out of the while loop and let it restart?
                     # Or just continue?
                     # Let's break and let the outer `while swap_improved` handle it.
                     break
             
             # After swaps, run reduction again to clean up
             changed = True
             while changed:
                 changed = False
                 to_remove = []
                 for c in code:
                     is_essential = False
                     for u in neighbors[c]:
                         if counts[u] == 1:
                             is_essential = True
                             break
                     if not is_essential:
                         to_remove.append(c)
                 
                 if to_remove:
                     for c in to_remove:
                         code.remove(c)
                         for u in neighbors[c]:
                             counts[u] -= 1
                     changed = True
                     
             if len(code) < best_len:
                 best_len = len(code)
                 best_code = code[:]

    if best_code is None:
        # Fallback: return a valid code (e.g., all words, or greedy result)
        # This shouldn't happen
        return [[0]*n]
        
    return [to_digits(c) for c in best_code]

if __name__ == "__main__":
    # Test with the specific task parameters
    q = 3
    n = 8
    radius = 1
    code = construct(q, n, radius)
    print(f"Code size: {len(code)}")
