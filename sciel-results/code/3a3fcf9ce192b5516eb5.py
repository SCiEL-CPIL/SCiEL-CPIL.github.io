import itertools
import random
import time
import heapq

def construct():
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute all K(2,2,2)s
    triple_to_ks = [[] for _ in range(num_triples)]
    k_to_triples = []
    
    matchings = [
        ((0,1),(2,3),(4,5)),
        ((0,1),(2,4),(3,5)),
        ((0,1),(2,5),(3,4)),
        ((0,2),(1,3),(4,5)),
        ((0,2),(1,4),(3,5)),
        ((0,2),(1,5),(3,4)),
        ((0,3),(1,2),(4,5)),
        ((0,3),(1,4),(2,5)),
        ((0,3),(1,5),(2,4)),
        ((0,4),(1,2),(3,5)),
        ((0,4),(1,3),(2,5)),
        ((0,4),(1,5),(2,3)),
        ((0,5),(1,2),(3,4)),
        ((0,5),(1,3),(2,4)),
        ((0,5),(1,4),(2,3))
    ]
    
    for subset in itertools.combinations(range(n), 6):
        for m in matchings:
            P1 = (subset[m[0][0]], subset[m[0][1]])
            P2 = (subset[m[1][0]], subset[m[1][1]])
            P3 = (subset[m[2][0]], subset[m[2][1]])
            
            k_triples = []
            for x in P1:
                for y in P2:
                    for z in P3:
                        t = tuple(sorted((x, y, z)))
                        k_triples.append(t)
            
            # Convert to indices
            k_triple_indices = [triple_to_idx[t] for t in k_triples]
            k_to_triples.append(k_triple_indices)
            
            for idx in k_triple_indices:
                triple_to_ks[idx].append(len(k_to_triples) - 1)

    num_ks = len(k_to_triples)

    def run_greedy_lsf_heap():
        counts = [0] * num_ks
        blocked = [0] * num_triples
        used = [False] * num_triples
        scores = [0] * num_triples
        
        # Heap entries: (score, idx)
        # We use lazy deletion
        heap = [(scores[i], i) for i in range(num_triples)]
        heapq.heapify(heap)
        
        result = []
        
        while heap:
            s, i = heapq.heappop(heap)
            
            # Check if this entry is still valid
            if used[i] or blocked[i] > 0 or scores[i] != s:
                continue
            
            # Select i
            result.append(all_triples[i])
            used[i] = True
            
            # Update counts and scores
            for k_idx in triple_to_ks[i]:
                old_count = counts[k_idx]
                counts[k_idx] += 1
                
                if old_count == 0:
                    # K becomes active
                    # Increment score of all other triples in this K
                    for t_idx in k_to_triples[k_idx]:
                        if not used[t_idx] and blocked[t_idx] == 0:
                            scores[t_idx] += 1
                            heapq.heappush(heap, (scores[t_idx], t_idx))
                
                if counts[k_idx] == 7:
                    # K becomes saturated, block the remaining triple
                    for t_idx in k_to_triples[k_idx]:
                        if not used[t_idx]:
                            blocked[t_idx] += 1
                            break

        return result

    best_result = []
    start_time = time.time()
    
    # Run as many restarts as possible
    while time.time() - start_time < 110:
        res = run_greedy_lsf_heap()
        if len(res) > len(best_result):
            best_result = res

    return best_result
