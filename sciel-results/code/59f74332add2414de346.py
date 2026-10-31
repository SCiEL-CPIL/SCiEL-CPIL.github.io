import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 140.0
    D = 11
    
    def run_greedy(weight, threshold_sq):
        vectors = []
        supports = list(itertools.combinations(range(D), weight))
        for s in supports:
            for signs in itertools.product([1, -1], repeat=weight):
                v = [0] * D
                for pos, sign in zip(s, signs):
                    v[pos] = sign
                vectors.append(v)
                
        if not vectors:
            return []
        
        vecs = np.array(vectors, dtype=np.int8)
        n_vecs = len(vecs)
        indices = np.arange(n_vecs)
        np.random.shuffle(indices)
        
        selected_vecs = []
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            v = vecs[idx]
            if not selected_vecs:
                selected_vecs.append(v)
                continue
            
            is_valid = True
            for u in selected_vecs:
                diff = u - v
                dist_sq = np.dot(diff, diff)
                if dist_sq < threshold_sq:
                    is_valid = False
                    break
            
            if is_valid:
                selected_vecs.append(v)
                
        return selected_vecs

    best_result = []
    
    # Try Weight 3
    res3 = run_greedy(3, 3)
    if len(res3) > len(best_result):
        best_result = res3
        
    if time.time() < deadline - 10:
        # Try Weight 4
        res4 = run_greedy(4, 4)
        if len(res4) > len(best_result):
            best_result = res4
            
    if best_result:
        return np.array(best_result, dtype=np.int8)
    else:
        # Fallback
        v = [0]*D
        v[0]=1; v[1]=1
        return np.array([v], dtype=np.int8)
