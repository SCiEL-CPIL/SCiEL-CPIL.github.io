import random
import time

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 54  # 6s safety margin for 60s limit
    
    best_size = float('inf')
    best_set = []
    
    def build_sum_set(A):
        """Build the set of all unordered sums a+b with a<=b for a,b in A."""
        S = set()
        for i, a in enumerate(A):
            S.add(2 * a)
            for b in A[i+1:]:
                S.add(a + b)
        return S
    
    def is_sidon(A):
        """Check if A is a Sidon set."""
        S = set()
        for i, a in enumerate(A):
            if 2 * a in S:
                return False
            S.add(2 * a)
            for b in A[i+1:]:
                s = a + b
                if s in S:
                    return False
                S.add(s)
        return True
    
    def is_maximal(A):
        """Check if A is inclusion-maximal Sidon in [1,N]."""
        if not A:
            return False
        A_set = set(A)
        S = build_sum_set(A)
        for x in range(1, N + 1):
            if x in A_set:
                continue
            if 2 * x in S:
                continue
            for a in A:
                if x + a in S:
                    break
            else:
                return False
        return True
    
    def greedy_construct(rng, strategy='random'):
        """Greedy construction of a Sidon set."""
        A = []
        A_set = set()
        S = set()
        
        while True:
            addable = []
            for x in range(1, N + 1):
                if x in A_set:
                    continue
                if 2 * x in S:
                    continue
                ok = True
                for a in A:
                    if x + a in S:
                        ok = False
                        break
                if ok:
                    addable.append(x)
            
            if not addable:
                break
            
            if strategy == 'smallest':
                x = min(addable)
            elif strategy == 'largest':
                x = max(addable)
            elif strategy == 'central':
                x = addable[len(addable) // 2]
            else:  # random
                x = rng.choice(addable)
            
            A.append(x)
            A_set.add(x)
            S.add(2 * x)
            for a in A[:-1]:
                S.add(x + a)
        
        return A
    
    def local_search_remove(A, rng, time_limit=2.0):
        """Try to remove elements while maintaining maximality."""
        A = list(A)
        A_set = set(A)
        local_start = time.time()
        
        while time.time() - local_start < time_limit:
            improved = False
            indices = list(range(len(A)))
            rng.shuffle(indices)
            
            for idx in indices:
                if time.time() - local_start > time_limit:
                    break
                if idx >= len(A):
                    continue
                
                candidate = A[idx]
                A_new = A[:idx] + A[idx+1:]
                
                if not A_new:
                    continue
                
                if is_maximal(A_new):
                    A = A_new
                    A_set = set(A)
                    improved = True
                    break
            
            if not improved:
                break
        
        return A
    
    def local_search_remove_pair(A, rng, time_limit=2.0):
        """Try to remove pairs of elements while maintaining maximality."""
        A = list(A)
        local_start = time.time()
        n = len(A)
        
        while time.time() - local_start < time_limit and len(A) >= 2:
            improved = False
            i_list = list(range(len(A)))
            rng.shuffle(i_list)
            
            for i in i_list:
                if time.time() - local_start > time_limit:
                    break
                if i >= len(A):
                    continue
                j_list = list(range(len(A)))
                j_list.remove(i)
                rng.shuffle(j_list)
                
                for j in j_list:
                    if time.time() - local_start > time_limit:
                        break
                    if j >= len(A):
                        continue
                    
                    # Build A without i and j
                    A_new = []
                    for k in range(len(A)):
                        if k != i and k != j:
                            A_new.append(A[k])
                    
                    if not A_new:
                        continue
                    
                    if is_maximal(A_new):
                        A = A_new
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return A
    
    # Main search loop
    for seed in range(5000):
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        
        strategies = ['random', 'smallest', 'largest', 'central']
        strategy = strategies[seed % len(strategies)]
        
        A = greedy_construct(rng, strategy)
        
        if not A:
            continue
        
        # Local search: try to remove elements
        A = local_search_remove(A, rng, time_limit=1.0)
        
        # Update best
        if len(A) < best_size:
            best_size = len(A)
            best_set = list(A)
            if best_size <= 12:
                A = local_search_remove(A, rng, time_limit=3.0)
                if len(A) < best_size:
                    best_size = len(A)
                    best_set = list(A)
    
    # Final polish: try to improve the best set found
    if best_set:
        rng = random.Random(42)
        best_set = local_search_remove(best_set, rng, time_limit=5.0)
        if len(best_set) < best_size:
            best_size = len(best_set)
        
        # Try pair removal as a final polish
        if best_size > 8:
            best_set = local_search_remove_pair(best_set, rng, time_limit=5.0)
            if len(best_set) < best_size:
                best_size = len(best_set)
    
    return best_set
