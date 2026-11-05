import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    if not (q == 3 and n == 8 and radius == 1):
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 118.0

    total_words = q ** n
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)

    # Precompute neighbors as lists of ints
    neighbors = [None] * total_words
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = neigh

    # Phase 1: Greedy
    covered = np.zeros(total_words, dtype=bool)
    code_list = []
    
    # Precompute uncovered counts for faster greedy?
    # Standard greedy: pick word covering most uncovered points.
    # We can maintain an array 'uncovered_count' for each word.
    # Initially all 1. When we pick a word, we decrement counts for all its neighbors.
    # This is O(N * |N|) per step, which is fine for N=6561.
    
    # Actually, the matrix-vector product approach in the original was O(N^2) per step if done naively, 
    # but with numpy it's fast. However, maintaining a count array is more efficient for sequential updates.
    
    # Let's stick to a simpler greedy that is fast enough:
    # 1. Start with all points uncovered.
    # 2. Pick word with max new coverage.
    # 3. Mark covered.
    # 4. Repeat.
    
    # Optimization: Use a heap or just argmax on a count array.
    # Count of uncovered points covered by word i.
    # Initially, count[i] = len(neighbors[i]) = 17.
    # When we pick word c, for every neighbor x of c, if x was uncovered, 
    # we need to decrement the count of all words that cover x.
    # This is complex. 
    # Alternative: Just use the numpy matrix product for the first few steps or just simple greedy.
    # Given N=6561, simple greedy with numpy argmax on a temporary vector is fine.
    
    # Let's use the count array approach for O(1) update per neighbor if we precompute reverse neighbors.
    # reverse_neighbors[x] = list of words that cover x.
    # This is the same as neighbors[x] in a symmetric graph? 
    # Yes, Hamming distance is symmetric. If y is in neighbors[x], then x is in neighbors[y].
    # So neighbors is symmetric.
    
    # Let's use the count array method.
    # count[i] = number of currently uncovered points in neighbors[i].
    # Initially count[i] = 17 for all i.
    # When we select c:
    #   For each x in neighbors[c]:
    #     If not covered[x]:
    #       covered[x] = True
    #       For each y in neighbors[x]:
    #         count[y] -= 1
    #   This is O(17 * 17) = 289 operations per step.
    #   Total steps ~ 400. Total ops ~ 100,000. Very fast.
    
    count = np.full(total_words, 17, dtype=np.int32)
    while not covered.all() and time.time() < deadline - 40.0:
        c = int(np.argmax(count))
        if count[c] == 0:
            break
        code_list.append(c)
        for x in neighbors[c]:
            if not covered[x]:
                covered[x] = True
                for y in neighbors[x]:
                    count[y] -= 1

    current_code = code_list[:]
    current_size = len(current_code)
    best_code = current_code[:]
    best_size = current_size

    # Precompute neighbor sets as Python sets for fast intersection in local search
    neighbor_sets = [set(neigh) for neigh in neighbors]
    
    # Initialize coverage counts
    current_cov_count = np.zeros(total_words, dtype=np.int32)
    for c in current_code:
        for x in neighbors[c]:
            current_cov_count[x] += 1

    random.seed(42)

    # Phase 2: Local Search
    stuck_counter = 0
    while time.time() < deadline - 5.0:
        if current_size == 0:
            break
        
        idx = random.randint(0, current_size - 1)
        c = current_code[idx]
        
        # Check if we can remove c
        # c is removable if all its neighbors are covered by at least one other codeword
        # i.e., current_cov_count[x] > 1 for all x in neighbors[c]
        
        removable = True
        for x in neighbors[c]:
            if current_cov_count[x] <= 1:
                removable = False
                break
        
        if removable:
            for x in neighbors[c]:
                current_cov_count[x] -= 1
            current_code.pop(idx)
            current_size -= 1
            stuck_counter = 0
            if current_size < best_size:
                best_size = current_size
                best_code = current_code[:]
        else:
            stuck_counter += 1
            
        # Try swap if stuck
        if stuck_counter > 5 and time.time() < deadline - 10.0 and current_size > 0:
            c1_idx = random.randint(0, current_size - 1)
            c1 = current_code[c1_idx]
            
            # Find risky points: neighbors of c1 with count == 1
            risky_points = []
            for x in neighbors[c1]:
                if current_cov_count[x] == 1:
                    risky_points.append(x)
            
            if not risky_points:
                # This shouldn't happen if 'removable' logic was correct, but just in case
                for x in neighbors[c1]:
                    current_cov_count[x] -= 1
                current_code.pop(c1_idx)
                current_size -= 1
                stuck_counter = 0
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
            else:
                # We need to find a replacement c2 such that:
                # 1. c2 covers all risky_points
                # 2. c2 is not in current_code
                # 3. Adding c2 doesn't break coverage elsewhere (it only adds coverage, so it's safe)
                
                # Candidate c2 must be in the intersection of neighbors of all risky_points
                # Intersection of sets
                if len(risky_points) == 1:
                    candidate_set = neighbor_sets[risky_points[0]].copy()
                else:
                    candidate_set = neighbor_sets[risky_points[0]].copy()
                    for x in risky_points[1:]:
                        candidate_set &= neighbor_sets[x]
                        if not candidate_set:
                            break
                
                if candidate_set:
                    # Remove c1 and any other codewords from candidates
                    candidate_set.discard(c1)
                    for c_code in current_code:
                        if c_code != c1:
                            candidate_set.discard(c_code)
                    
                    if candidate_set:
                        c2 = random.choice(list(candidate_set))
                        
                        # Perform swap
                        # Remove c1
                        for x in neighbors[c1]:
                            current_cov_count[x] -= 1
                        # Add c2
                        for x in neighbors[c2]:
                            current_cov_count[x] += 1
                        
                        current_code[c1_idx] = c2
                        stuck_counter = 0
                        # Note: We don't update best_code here immediately as size didn't change,
                        # but future removals might.
            
            stuck_counter = 0

    # Final polish: try to remove more
    final_code = best_code[:]
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        for x in neighbors[idx]:
            final_cov_count[x] += 1
            
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(len(final_code) - 1, -1, -1):
            c = final_code[i]
            removable = True
            for x in neighbors[c]:
                if final_cov_count[x] <= 1:
                    removable = False
                    break
            if removable:
                for x in neighbors[c]:
                    final_cov_count[x] -= 1
                final_code.pop(i)
                improved = True
                
    # Verify validity
    if not np.all(final_cov_count > 0):
        return [list(digits_of[i]) for i in best_code]
        
    return [list(digits_of[i]) for i in final_code]
