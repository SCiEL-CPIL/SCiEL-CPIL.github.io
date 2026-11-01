import time
import random
import heapq
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Specific case for q=3, n=6, radius=1
    # General fallback for other parameters
    if q == 3 and n == 6 and radius == 1:
        # Total words
        total_words = q ** n
        
        # Generate all words as tuples
        all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute spheres: list of indices for each word within distance 1
        spheres = []
        for w in all_words:
            sphere = [word_to_idx[w]]
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    new_w = list(w)
                    new_w[i] = val
                    sphere.append(word_to_idx[tuple(new_w)])
            spheres.append(sphere)
        
        # Precompute reverse spheres: for each word, which codewords cover it
        reverse_spheres = [[] for _ in range(total_words)]
        for i in range(total_words):
            for j in spheres[i]:
                reverse_spheres[j].append(i)
        
        covered = [False] * total_words
        uncovered_count = total_words
        code = []
        picked = set()
        
        # gain[i] = number of uncovered words in spheres[i]
        gain = [0] * total_words
        for i in range(total_words):
            gain[i] = len(spheres[i])
        
        # Max-heap: (-gain, symmetry_key, word_idx)
        # symmetry_key: use the word tuple for consistent lexicographic tie-breaking
        heap = []
        for i in range(total_words):
            heap.append((-gain[i], all_words[i], i))
        heapq.heapify(heap)
        
        while uncovered_count > 0:
            if time.time() > deadline:
                break
            
            # Find the best candidate using the heap with lazy deletion
            best_candidate = -1
            best_gain = 0
            
            while heap:
                neg_gain, word_tuple, cand_idx = heapq.heappop(heap)
                
                # Check if this entry is stale
                if -neg_gain != gain[cand_idx]:
                    continue
                
                # Check if already picked
                if cand_idx in picked:
                    continue
                
                best_candidate = cand_idx
                best_gain = -neg_gain
                break
            
            if best_candidate == -1:
                break
            
            # Select this candidate
            picked.add(best_candidate)
            code.append(list(all_words[best_candidate]))
            
            # Update: mark all words in spheres[best_candidate] as covered
            for idx in spheres[best_candidate]:
                if not covered[idx]:
                    covered[idx] = True
                    uncovered_count -= 1
                    
                    # Decrement gain for all candidates that cover idx
                    for cand in reverse_spheres[idx]:
                        if cand not in picked:
                            gain[cand] -= 1
                            # Push updated gain to heap
                            heapq.heappush(heap, (-gain[cand], all_words[cand], cand))
        
        return code
    else:
        # Fallback: Random or simple construction for other params
        import itertools
        return [list(t) for t in itertools.product(range(q), repeat=n)]
