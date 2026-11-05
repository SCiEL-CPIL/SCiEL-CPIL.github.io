import numpy as np
import time
import random

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # Target n
    # 34 is the known bound, but finding it from scratch is hard.
    # 25-30 is achievable with local search.
    n = 30
    
    # Precompute triple indices
    triples = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                triples.append((a, b, c))
    num_triples = len(triples)
    
    # Map (a,b,c) to index
    triple_index = {}
    for i, t in enumerate(triples):
        triple_index[t] = i
        
    # Precompute 4-sets and their triple indices
    red_sets = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    idxs = [
                        triple_index[(a,b,c)],
                        triple_index[(a,b,d)],
                        triple_index[(a,c,d)],
                        triple_index[(b,c,d)]
                    ]
                    red_sets.append(idxs)
    red_indices = np.array(red_sets, dtype=np.int32)
    
    # Precompute 5-sets and their triple indices
    blue_sets = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    for e in range(d+1, n):
                        idxs = [
                            triple_index[(a,b,c)],
                            triple_index[(a,b,d)],
                            triple_index[(a,b,e)],
                            triple_index[(a,c,d)],
                            triple_index[(a,c,e)],
                            triple_index[(a,d,e)],
                            triple_index[(b,c,d)],
                            triple_index[(b,c,e)],
                            triple_index[(b,d,e)],
                            triple_index[(c,d,e)]
                        ]
                        blue_sets.append(idxs)
    blue_indices = np.array(blue_sets, dtype=np.int32)
    
    # Local search
    for restart in range(50):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize colors randomly
        colors = np.random.randint(0, 2, num_triples, dtype=np.int8)
        
        for step in range(2000):
            if time.time() - start_time > time_limit:
                break
                
            # Check violations
            red_sums = np.sum(colors[red_indices], axis=1)
            blue_sums = np.sum(colors[blue_indices], axis=1)
            
            red_viol_idx = np.where(red_sums == 0)[0]
            blue_viol_idx = np.where(blue_sums == 10)[0]
            
            num_red_viol = len(red_viol_idx)
            num_blue_viol = len(blue_viol_idx)
            
            if num_red_viol == 0 and num_blue_viol == 0:
                return {'n': n, 'colors': colors.tolist()}
            
            # Pick a violation to fix
            if num_red_viol > 0 and (num_blue_viol == 0 or random.random() < 0.5):
                set_idx = red_viol_idx[random.randint(0, num_red_viol - 1)]
                t_idx = red_indices[set_idx][random.randint(0, 3)]
                colors[t_idx] = 1
            else:
                set_idx = blue_viol_idx[random.randint(0, num_blue_viol - 1)]
                t_idx = blue_indices[set_idx][random.randint(0, 9)]
                colors[t_idx] = 0
                
            # Random perturbation
            if step % 50 == 0:
                colors[random.randint(0, num_triples - 1)] = random.randint(0, 1)

    # Fallback to n=25
    n = 25
    triples = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                triples.append((a, b, c))
    num_triples = len(triples)
    triple_index = {}
    for i, t in enumerate(triples):
        triple_index[t] = i
        
    red_sets = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    idxs = [
                        triple_index[(a,b,c)],
                        triple_index[(a,b,d)],
                        triple_index[(a,c,d)],
                        triple_index[(b,c,d)]
                    ]
                    red_sets.append(idxs)
    red_indices = np.array(red_sets, dtype=np.int32)
    
    blue_sets = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    for e in range(d+1, n):
                        idxs = [
                            triple_index[(a,b,c)],
                            triple_index[(a,b,d)],
                            triple_index[(a,b,e)],
                            triple_index[(a,c,d)],
                            triple_index[(a,c,e)],
                            triple_index[(a,d,e)],
                            triple_index[(b,c,d)],
                            triple_index[(b,c,e)],
                            triple_index[(b,d,e)],
                            triple_index[(c,d,e)]
                        ]
                        blue_sets.append(idxs)
    blue_indices = np.array(blue_sets, dtype=np.int32)
    
    for restart in range(50):
        if time.time() - start_time > time_limit:
            break
            
        colors = np.random.randint(0, 2, num_triples, dtype=np.int8)
        
        for step in range(2000):
            if time.time() - start_time > time_limit:
                break
                
            red_sums = np.sum(colors[red_indices], axis=1)
            blue_sums = np.sum(colors[blue_indices], axis=1)
            
            red_viol_idx = np.where(red_sums == 0)[0]
            blue_viol_idx = np.where(blue_sums == 10)[0]
            
            num_red_viol = len(red_viol_idx)
            num_blue_viol = len(blue_viol_idx)
            
            if num_red_viol == 0 and num_blue_viol == 0:
                return {'n': n, 'colors': colors.tolist()}
            
            if num_red_viol > 0 and (num_blue_viol == 0 or random.random() < 0.5):
                set_idx = red_viol_idx[random.randint(0, num_red_viol - 1)]
                t_idx = red_indices[set_idx][random.randint(0, 3)]
                colors[t_idx] = 1
            else:
                set_idx = blue_viol_idx[random.randint(0, num_blue_viol - 1)]
                t_idx = blue_indices[set_idx][random.randint(0, 9)]
                colors[t_idx] = 0
                
            if step % 50 == 0:
                colors[random.randint(0, num_triples - 1)] = random.randint(0, 1)

    # Fallback to n=20
    n = 20
    triples = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                triples.append((a, b, c))
    num_triples = len(triples)
    triple_index = {}
    for i, t in enumerate(triples):
        triple_index[t] = i
        
    red_sets = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    idxs = [
                        triple_index[(a,b,c)],
                        triple_index[(a,b,d)],
                        triple_index[(a,c,d)],
                        triple_index[(b,c,d)]
                    ]
                    red_sets.append(idxs)
    red_indices = np.array(red_sets, dtype=np.int32)
    
    blue_sets = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    for e in range(d+1, n):
                        idxs = [
                            triple_index[(a,b,c)],
                            triple_index[(a,b,d)],
                            triple_index[(a,b,e)],
                            triple_index[(a,c,d)],
                            triple_index[(a,c,e)],
                            triple_index[(a,d,e)],
                            triple_index[(b,c,d)],
                            triple_index[(b,c,e)],
                            triple_index[(b,d,e)],
                            triple_index[(c,d,e)]
                        ]
                        blue_sets.append(idxs)
    blue_indices = np.array(blue_sets, dtype=np.int32)
    
    for restart in range(100):
        if time.time() - start_time > time_limit:
            break
            
        colors = np.random.randint(0, 2, num_triples, dtype=np.int8)
        
        for step in range(2000):
            if time.time() - start_time > time_limit:
                break
                
            red_sums = np.sum(colors[red_indices], axis=1)
            blue_sums = np.sum(colors[blue_indices], axis=1)
            
            red_viol_idx = np.where(red_sums == 0)[0]
            blue_viol_idx = np.where(blue_sums == 10)[0]
            
            num_red_viol = len(red_viol_idx)
            num_blue_viol = len(blue_viol_idx)
            
            if num_red_viol == 0 and num_blue_viol == 0:
                return {'n': n, 'colors': colors.tolist()}
            
            if num_red_viol > 0 and (num_blue_viol == 0 or random.random() < 0.5):
                set_idx = red_viol_idx[random.randint(0, num_red_viol - 1)]
                t_idx = red_indices[set_idx][random.randint(0, 3)]
                colors[t_idx] = 1
            else:
                set_idx = blue_viol_idx[random.randint(0, num_blue_viol - 1)]
                t_idx = blue_indices[set_idx][random.randint(0, 9)]
                colors[t_idx] = 0
                
            if step % 50 == 0:
                colors[random.randint(0, num_triples - 1)] = random.randint(0, 1)

    # Final fallback: n=9
    n = 9
    triples = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                triples.append((a, b, c))
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    return {'n': n, 'colors': colors}
