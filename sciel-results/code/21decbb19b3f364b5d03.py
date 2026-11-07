import time
import random
import math

P = {'n': 12, 'm': 40}

def construct():
    deadline = time.time() + 85
    n = P['n']
    m = P['m']
    N = 1 << n
    
    # Precompute comparability matrix as a list of lists of booleans
    # comp_mat[a][b] = True if a and b are comparable (one is subset of other)
    # This is N x N = 4096 x 4096 = ~16.7M booleans, manageable
    comp_mat = [[False] * N for _ in range(N)]
    for a in range(N):
        for b in range(N):
            if (a & b) == a or (a & b) == b:
                comp_mat[a][b] = True
    
    # Precompute for each mask, the list of comparable masks
    # And the count of comparable masks
    comp_count = [0] * N
    for a in range(N):
        for b in range(N):
            if comp_mat[a][b]:
                comp_count[a] += 1
    
    def count_comparable(family):
        count = 0
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                if comp_mat[family[i]][family[j]]:
                    count += 1
        return count
    
    def local_search(family, max_iters, deadline):
        # family is a list of masks
        # comp_in_F[y] = number of sets in family comparable to y
        comp_in_F = [0] * N
        
        # Initialize comp_in_F efficiently
        for z in family:
            for y in range(N):
                if comp_mat[y][z]:
                    comp_in_F[y] += 1
            
        current_score = count_comparable(family)
        best_score = current_score
        best_family = family[:]
        
        fam_set = set(family)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            improved = False
            for i in range(m):
                x = family[i]
                k_x = comp_in_F[x] - 1
                
                best_y = -1
                best_val = -1
                
                for y in range(N):
                    if y in fam_set:
                        continue
                    val = comp_in_F[y]
                    if comp_mat[y][x]:
                        val -= 1
                    
                    if val > best_val:
                        best_val = val
                        best_y = y
                
                if best_y == -1:
                    continue
                
                if best_val > k_x:
                    # Update comp_in_F
                    for z in range(N):
                        delta = 0
                        if comp_mat[z][best_y]:
                            delta += 1
                        if comp_mat[z][x]:
                            delta -= 1
                        if delta != 0:
                            comp_in_F[z] += delta
                    
                    fam_set.remove(x)
                    fam_set.add(best_y)
                    family[i] = best_y
                    current_score += best_val - k_x
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_family = family[:]
                        
                    improved = True
            
            if not improved:
                break
                
        return best_family, best_score
    
    def simulated_annealing(family, deadline):
        # SA to escape local minima
        comp_in_F = [0] * N
        for z in family:
            for y in range(N):
                if comp_mat[y][z]:
                    comp_in_F[y] += 1
        
        current_score = count_comparable(family)
        best_score = current_score
        best_family = family[:]
        
        fam_set = set(family)
        
        T = 5.0
        cooling = 0.995
        min_T = 0.01
        
        while T > min_T and time.time() < deadline - 2:
            T *= cooling
            
            # Pick a random position
            i = random.randint(0, m - 1)
            x = family[i]
            
            # Pick a random candidate not in family
            candidates = [y for y in range(N) if y not in fam_set]
            if not candidates:
                continue
            y = random.choice(candidates)
            
            if y == x:
                continue
            
            # Compute score change
            k_x = comp_in_F[x] - 1
            val_y = comp_in_F[y]
            if comp_mat[y][x]:
                val_y -= 1
            
            delta = val_y - k_x
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                # Accept the move
                for z in range(N):
                    d = 0
                    if comp_mat[z][y]:
                        d += 1
                    if comp_mat[z][x]:
                        d -= 1
                    if d != 0:
                        comp_in_F[z] += d
                
                fam_set.remove(x)
                fam_set.add(y)
                family[i] = y
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_family = family[:]
        
        return best_family, best_score
    
    best_overall = None
    best_score_overall = 0
    
    # Multiple restarts with local search
    for restart in range(15):
        if time.time() > deadline - 15:
            break
        
        random.seed(42 + restart)
        family = random.sample(range(N), m)
        
        fam, score = local_search(family, 5000, deadline)
        
        if score > best_score_overall:
            best_score_overall = score
            best_overall = fam[:]
    
    # Run SA from the best solution found so far
    if best_overall is not None and time.time() < deadline - 10:
        fam, score = simulated_annealing(best_overall[:], deadline)
        if score > best_score_overall:
            best_score_overall = score
            best_overall = fam[:]
    
    # Try chain-based initialization
    if time.time() < deadline - 15:
        chain = [0]
        cur = 0
        for b in range(n):
            cur |= (1 << b)
            chain.append(cur)
        remaining = [x for x in range(N) if x not in chain]
        random.seed(999)
        extra = random.sample(remaining, m - len(chain))
        family = chain + extra
        fam, score = local_search(family, 5000, deadline)
        if score > best_score_overall:
            best_score_overall = score
            best_overall = fam[:]
    
    # Final SA pass from best
    if best_overall is not None and time.time() < deadline - 5:
        fam, score = simulated_annealing(best_overall[:], deadline)
        if score > best_score_overall:
            best_score_overall = score
            best_overall = fam[:]
    
    if best_overall is None:
        best_overall = list(range(m))
    
    result = []
    for mask in best_overall:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
