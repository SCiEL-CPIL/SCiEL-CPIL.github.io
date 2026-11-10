import time
import itertools
import random
import sys

def construct(q, n, radius):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4 and radius == 1:
            # Simple recursive or product construction fallback
            if n == 4:
                return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            else:
                # Direct product with tetracode for n=4, then extend?
                # Just return all words if too complex, or a large code
                # For this task, we focus on 7,1
                pass
        
        # Generic fallback: return all words (worst case, but valid)
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific logic for q=3, n=7, radius=1
    total_words = q ** n
    neighbors = [[] for _ in range(total_words)]
    
    # Precompute neighbors for all words
    # word is an integer 0..2186
    # Convert to base-3 digits
    def int_to_word(val):
        word = [0] * n
        for i in range(n):
            word[i] = val % q
            val //= q
        return word

    def word_to_int(word):
        result = 0
        for w in word:
            result = result * q + w
        return result

    # Generate all words and their neighbors
    for i in range(total_words):
        w = int_to_word(i)
        neighbors[i].append(i)
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    new_w = w.copy()
                    new_w[pos] = new_val
                    neighbors[i].append(word_to_int(new_w))
        # Remove duplicates if any (shouldn't be any for radius 1 in q>2)
        neighbors[i] = list(set(neighbors[i]))

    # 1. Greedy Set Cover
    covered = [False] * total_words
    code = []
    
    # To speed up greedy, we can maintain a list of uncovered words?
    # Or just iterate all candidates each time.
    # 2187 candidates * 15 neighbors * ~180 steps = ~6M ops. OK.
    
    for _ in range(200): # Max steps safety
        if time.time() > deadline - 5:
            break
            
        best_candidate = -1
        best_count = -1
        
        # Optimization: Only consider candidates that cover at least one uncovered word?
        # Hard to know without checking.
        
        for c in range(total_words):
            # Skip if already in code? 
            # Actually, adding a duplicate is useless.
            if c in code:
                continue
                
            count = 0
            for nb in neighbors[c]:
                if not covered[nb]:
                    count += 1
            if count > best_count:
                best_count = count
                best_candidate = c
        
        if best_count <= 0:
            break
        
        # Add best_candidate
        code.append(best_candidate)
        for nb in neighbors[best_candidate]:
            covered[nb] = True
            
    # 2. Local Search / Optimization
    # Iterate until no improvement
    improved = True
    while improved:
        if time.time() > deadline - 2:
            break
        improved = False
        
        # --- Deletion Step ---
        # Check if any codeword is redundant
        code_set = set(code)
        to_remove = []
        
        # Precompute coverage by others for each word?
        # For each c in code, let Unique(c) = B(c) \ Union(B(c') for c' in code\{c})
        # If Unique(c) is empty, c is redundant.
        
        # Efficient check:
        # For each word w, let Coverers(w) be the set of codewords in code that cover w.
        # If for some c, there is no w such that Coverers(w) == {c}, then c is redundant?
        # No, if Coverers(w) is non-empty for all w, and for each c there exists w with c in Coverers(w), 
        # it doesn't mean c is essential.
        # Essential if there is a w covered ONLY by c.
        
        # Let's compute for each w in space, which codewords cover it.
        # Map: word -> list of indices in code that cover it.
        # Since code size is ~180, we can do this.
        
        coverers = [[] for _ in range(total_words)]
        for idx, c in enumerate(code):
            for nb in neighbors[c]:
                coverers[nb].append(idx)
        
        # Check redundancy
        for idx, c in enumerate(code):
            # If there is no word w such that coverers[w] == [idx], then c is redundant
            # Wait, if coverers[w] contains idx and others, c is not unique for w.
            # We need to check if c is the *only* coverer for *any* w.
            is_redundant = True
            for nb in neighbors[c]:
                if len(coverers[nb]) == 1 and coverers[nb][0] == idx:
                    is_redundant = False
                    break
            
            if is_redundant:
                to_remove.append(idx)
                improved = True
        
        if to_remove:
            # Remove from end to start to preserve indices
            for idx in sorted(to_remove, reverse=True):
                code.pop(idx)
            # Recalculate coverers if we continue?
            # The loop restarts, so coverers will be recalculated.
            continue # Restart loop to recalculate coverers
        
        # --- 2-for-1 Swap Step ---
        # Try to replace c_i, c_j with c_k
        # Condition: B(c_k) covers all words that were uniquely covered by c_i or c_j relative to the rest.
        
        # Recalculate coverers as above (already done if no deletion)
        # Actually, if we didn't delete, coverers is valid.
        
        n_code = len(code)
        for i in range(n_code):
            if time.time() > deadline - 2:
                break
            for j in range(i + 1, n_code):
                # Words covered by i or j
                # We need to find words w such that {i, j} are the ONLY coverers?
                # No, we remove i and j.
                # Words that become uncovered are those w where coverers[w] is a subset of {i, j} and non-empty.
                # Specifically, if coverers[w] == [i], [j], or [i, j], then w is lost if we remove i, j.
                # Let Lost = { w | coverers[w] is a non-empty subset of {i, j} }
                
                lost_words = []
                for w in range(total_words):
                    covs = coverers[w]
                    if len(covs) == 0:
                        continue
                    if len(covs) == 1:
                        if covs[0] == i or covs[0] == j:
                            lost_words.append(w)
                    elif len(covs) == 2:
                        if (covs[0] == i and covs[1] == j) or (covs[0] == j and covs[1] == i):
                            lost_words.append(w)
                
                if not lost_words:
                    # This is a deletion of 2, which should have been caught?
                    # Not necessarily, if they cover disjoint unique sets?
                    # If lost_words is empty, it means every word covered by i or j is also covered by someone else.
                    # So i and j are BOTH redundant.
                    # The deletion step should have caught this.
                    # But maybe not if they were mutually dependent?
                    # If i covers w1 uniquely, and j covers w2 uniquely.
                    # Then for w1, coverers is [i]. For w2, coverers is [j].
                    # So lost_words would be [w1, w2].
                    # So lost_words is empty implies both are redundant.
                    # We can just delete them?
                    # But our deletion step checks one by one.
                    # If we delete i, j might become redundant.
                    # Let's just add a dummy candidate?
                    # Or skip. The deletion step will handle it in next iteration.
                    continue
                
                # We need a new codeword k such that B(k) covers all lost_words.
                # Candidate k must be in intersection of neighbors of all lost_words.
                
                # Intersection of neighbors
                # Start with neighbors of first lost word
                candidates = set(neighbors[lost_words[0]])
                for w in lost_words[1:]:
                    candidates.intersection_update(neighbors[w])
                    if not candidates:
                        break
                
                if candidates:
                    # Pick one candidate
                    # Prefer one not in code?
                    # Or any.
                    new_c = next(iter(candidates))
                    
                    # Verify: Does replacing i, j with new_c maintain coverage?
                    # We know new_c covers lost_words.
                    # We need to ensure no other word is uncovered.
                    # The only words at risk are those covered by i or j.
                    # Any word covered by i or j is either in lost_words (uncovered by others) or covered by others.
                    # If covered by others, it remains covered.
                    # If in lost_words, it is covered by new_c.
                    # So yes, it works.
                    
                    # Perform swap
                    # Remove i and j
                    # Add new_c
                    
                    # We must be careful with indices.
                    # Let's create a new code list.
                    new_code = []
                    for idx, c in enumerate(code):
                        if idx != i and idx != j:
                            new_code.append(c)
                    new_code.append(new_c)
                    
                    # Check if new_code is valid? 
                    # We argued it is.
                    # But let's be safe and verify coverage? 
                    # Verification is O(2187 * 15) = 32k. 
                    # Doing this for every swap attempt might be slow if many attempts.
                    # But we only do it if we find a candidate.
                    # Number of pairs is ~16k. 
                    # If most have no candidate, it's fast.
                    # If many have candidates, we verify.
                    
                    # Let's do a quick check:
                    # Is new_c already in new_code?
                    if new_c in new_code:
                        # We added a duplicate. 
                        # Then we effectively removed i and j and added a duplicate of an existing word.
                        # Size decreased by 1.
                        # But is it valid?
                        # If new_c was already in code, its coverage was already counted.
                        # We removed i and j.
                        # The words covered by new_c were already covered by new_c.
                        # The lost_words were covered by new_c.
                        # So yes.
                        pass
                    else:
                        # Size decreased by 1.
                        pass
                    
                    code = new_code
                    improved = True
                    break # Restart loop to recalculate coverers
            if improved:
                break
                
    # Convert code to list of lists
    return [int_to_word(c) for c in code]

if __name__ == "__main__":
    # Test
    code = construct(3, 7, 1)
    print(f"Code size: {len(code)}")
