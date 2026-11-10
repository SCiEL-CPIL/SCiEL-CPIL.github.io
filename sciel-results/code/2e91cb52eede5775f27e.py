import itertools
import random
from collections import defaultdict

def construct(q, n, radius):
    # Total number of words
    total_words = q ** n
    
    # Precompute neighbors for each word
    # neighbors[w] is a list of words within Hamming distance <= radius from w
    neighbors = []
    # Helper to convert int to base-q list
    def int_to_base_q(x, n, q):
        digits = []
        for _ in range(n):
            digits.append(x % q)
            x //= q
        return digits
    
    # Helper to convert base-q list to int
    def base_q_to_int(digits, q):
        x = 0
        for i, d in enumerate(reversed(digits)):
            x += d * (q ** i)
        return x

    # Generate all words
    words = list(range(total_words))
    
    # Precompute neighbors
    # For efficiency, we can generate neighbors on the fly or precompute.
    # Given total_words is small (e.g., 6561), precomputation is fine.
    # However, doing it for every call might be slow if n is large, but here n is fixed in the task context usually.
    # Let's precompute a list of neighbors for each word index.
    
    # Optimization: Instead of storing full lists, we can just calculate distance on the fly if needed, 
    # but for the greedy step, we need to iterate neighbors frequently.
    
    # Let's precompute neighbors for all words
    all_neighbors = [[] for _ in range(total_words)]
    for w in words:
        dw = int_to_base_q(w, n, q)
        # Center itself
        all_neighbors[w].append(w)
        # Neighbors at distance 1
        for i in range(n):
            for d in range(q):
                if d == dw[i]:
                    continue
                nd = dw[:]
                nd[i] = d
                nw = base_q_to_int(nd, q)
                all_neighbors[w].append(nw)
        
        # If radius > 1, we would need to go deeper. 
        # The task specifies radius, but typically R=1 for this specific 8-match pool problem.
        # The prompt says "radius" is passed. We must handle general radius.
        # However, for R > 1, the number of neighbors grows significantly.
        # For R=1, count is 1 + n*(q-1).
        # For R=2, it's larger.
        # Given the constraint "finish within 120s" and n=8, q=3, R=1 is the likely case.
        # If R is larger, the greedy approach might need adjustment, but let's assume R=1 or small R.
        
        # To be safe for R > 1, we should generate all words within distance R.
        # But that's expensive to precompute for all w if R is large.
        # Let's stick to the assumption that R is small (1 or 2).
        # Actually, the prompt says "checked over the whole space".
        # Let's refine the neighbor generation to be correct for any R, but efficient.
        
    # Re-do neighbor generation correctly for any R
    all_neighbors = [[] for _ in range(total_words)]
    for w in words:
        dw = int_to_base_q(w, n, q)
        # Use recursive generation or itertools.product for positions to change
        # For R=1, we did it above.
        # For general R, we can use a BFS or just iterate all words and check distance?
        # Iterating all words (6561) for each w (6561) is 43M operations. Might be too slow in Python.
        # Better to generate neighbors structurally.
        
        # Let's use a set to avoid duplicates
        neigh_set = set()
        # Generate all combinations of positions to change
        # This is complex for arbitrary R.
        # Given the specific task "Football pool problem for 8 matches", R is almost certainly 1.
        # The best known code 486 is for R=1.
        # I will implement the R=1 case specifically for speed, and fall back to a slower method if R != 1.
        
        if radius == 1:
            neigh_set.add(w)
            for i in range(n):
                for d in range(q):
                    if d == dw[i]:
                        continue
                    nd = dw[:]
                    nd[i] = d
                    nw = base_q_to_int(nd, q)
                    neigh_set.add(nw)
        else:
            # Fallback: check all words
            for u in words:
                du = int_to_base_q(u, n, q)
                dist = sum(1 for i in range(n) if dw[i] != du[i])
                if dist <= radius:
                    neigh_set.add(u)
                    
        all_neighbors[w] = list(neigh_set)

    # Greedy Set Cover
    uncovered = set(words)
    code = []
    
    # counts[c] = number of uncovered words covered by c
    counts = [0] * total_words
    
    # Initialize counts
    # For each uncovered word w, increment counts for all c that cover w
    # This is equivalent to: for each c, count how many of its neighbors are in uncovered
    # Let's just compute it directly for the first step, and update dynamically.
    
    # To speed up the "find max" step, we can maintain a list of candidates.
    # But with 6561 candidates, a linear scan is O(6561).
    # Steps ~ 500. 500 * 6561 ~ 3M operations. Very fast.
    
    # Initialize counts
    for c in range(total_words):
        # Count how many neighbors of c are in uncovered
        # Since initially all are uncovered, this is just len(all_neighbors[c])
        counts[c] = len(all_neighbors[c])

    while uncovered:
        # Find c with max counts[c]
        best_c = -1
        max_count = -1
        for c in range(total_words):
            if counts[c] > max_count:
                max_count = counts[c]
                best_c = c
                
        if best_c == -1 or max_count == 0:
            break # Should not happen if uncovered is not empty
            
        code.append(best_c)
        
        # Update uncovered and counts
        # Remove neighbors of best_c from uncovered
        to_remove = []
        for w in all_neighbors[best_c]:
            if w in uncovered:
                to_remove.append(w)
                
        for w in to_remove:
            uncovered.remove(w)
            
        # Update counts for all candidates c' that cover any of the removed words
        # For each removed word w, find all c' such that w in all_neighbors[c']
        # This requires a reverse mapping: word_to_centers[w] = list of c' covering w
        # Let's precompute this reverse mapping to make updates fast.
        
    # Precompute reverse mapping
    word_to_centers = [[] for _ in range(total_words)]
    for c in range(total_words):
        for w in all_neighbors[c]:
            word_to_centers[w].append(c)

    # Re-run greedy with dynamic updates
    uncovered = set(words)
    code = []
    counts = [0] * total_words
    
    # Initial counts
    for c in range(total_words):
        counts[c] = len(all_neighbors[c]) # All are uncovered initially

    while uncovered:
        best_c = -1
        max_count = -1
        # To optimize, we could use a heap, but linear scan is fine for 6561
        for c in range(total_words):
            if counts[c] > max_count:
                max_count = counts[c]
                best_c = c
                
        if best_c == -1:
            break

        code.append(best_c)
        
        # Determine which words are covered by best_c and were uncovered
        newly_covered = []
        for w in all_neighbors[best_c]:
            if w in uncovered:
                newly_covered.append(w)
                uncovered.remove(w)
                
        # Update counts
        # For each newly covered word w, decrement counts for all c' that cover w
        for w in newly_covered:
            for c_prime in word_to_centers[w]:
                counts[c_prime] -= 1
                # If counts becomes 0, we could remove it from consideration, 
                # but linear scan handles it.

    # Local Optimization (Swaps)
    # We have a valid code 'code'.
    # We want to minimize len(code).
    # 1. Try to remove codewords that are redundant (cover no unique words).
    # 2. Try to swap codewords to enable further removals.
    
    # Compute cover counts for each word in the space
    # cover_count[w] = number of codewords in code covering w
    cover_count = [0] * total_words
    code_set = set(code)
    
    for c in code:
        for w in all_neighbors[c]:
            cover_count[w] += 1
            
    # Identify private words for each codeword
    # private_words[c] = set of w such that cover_count[w] == 1 and c covers w
    private_words = {c: set() for c in code}
    
    for c in code:
        for w in all_neighbors[c]:
            if cover_count[w] == 1:
                private_words[c].add(w)
                
    improved = True
    while improved:
        improved = False
        # Try to remove
        new_code = []
        for c in code:
            if len(private_words[c]) == 0:
                # Can remove c
                # Update cover_count and private_words for remaining code
                # Remove c from code_set
                code_set.remove(c)
                
                # Decrement cover_count for neighbors of c
                for w in all_neighbors[c]:
                    cover_count[w] -= 1
                    # If cover_count[w] becomes 1, it becomes private to the remaining coder
                    # We need to update private_words for the other coders
                    # This is tricky to do incrementally without rebuilding.
                    # Given the small size, let's rebuild private_words if we remove a word.
                    pass
                improved = True
                # Restart the loop to recalculate private words
                break
            else:
                new_code.append(c)
                
        if improved:
            code = new_code
            # Recalculate cover_count and private_words from scratch
            cover_count = [0] * total_words
            code_set = set(code)
            for c in code:
                for w in all_neighbors[c]:
                    cover_count[w] += 1
            private_words = {c: set() for c in code}
            for c in code:
                for w in all_neighbors[c]:
                    if cover_count[w] == 1:
                        private_words[c].add(w)
            continue
            
        # Try to swap
        # For each c in code, if private_words[c] is not empty, try to find c_new
        # such that private_words[c] is subset of all_neighbors[c_new]
        # and c_new is not in code_set.
        
        for c in code:
            if len(private_words[c]) == 0:
                continue
                
            p_set = private_words[c]
            found_swap = False
            # Iterate through all possible c_new
            # To speed up, we can check only c_new that cover at least one word in p_set
            # Candidates are in union of word_to_centers[w] for w in p_set
            candidate_new = set()
            for w in p_set:
                candidate_new.update(word_to_centers[w])
            
            for c_new in candidate_new:
                if c_new in code_set:
                    continue
                # Check if c_new covers all words in p_set
                # i.e., p_set is subset of all_neighbors[c_new]
                # Since p_set is small, we can check each w in p_set
                covers_all = True
                for w in p_set:
                    if w not in all_neighbors[c_new]:
                        covers_all = False
                        break
                if covers_all:
                    # Perform swap
                    code_set.remove(c)
                    code_set.add(c_new)
                    
                    # Replace c with c_new in code list
                    code[code.index(c)] = c_new
                    
                    # Update cover_count
                    # Decrement for neighbors of c
                    for w in all_neighbors[c]:
                        cover_count[w] -= 1
                    # Increment for neighbors of c_new
                    for w in all_neighbors[c_new]:
                        cover_count[w] += 1
                        
                    # Recalculate private_words
                    private_words = {c: set() for c in code}
                    for c_curr in code:
                        for w in all_neighbors[c_curr]:
                            if cover_count[w] == 1:
                                private_words[c_curr].add(w)
                                
                    improved = True
                    found_swap = True
                    break
                    
            if found_swap:
                break

    # Convert integer codewords back to lists of integers
    result = []
    for c in code:
        result.append(int_to_base_q(c, n, q))
        
    return result
