"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def build_conflict_graph(S):
        """Build adjacency list where edge (i,j) means i+j is also in S."""
        S_set = set(S)
        n = len(S)
        idx_map = {x: i for i, x in enumerate(S)}
        adj = [set() for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                s = S[i] + S[j]
                if s in S_set:
                    k = idx_map[s]
                    # Conflict: including both i and j forbids k
                    # But for MIS on "sum-free", we need: if i,j in set, then i+j not in set
                    # So conflict is between {i,j} and {k}
                    # This is a 3-uniform hypergraph, not a graph!
                    # Alternative: model as graph where vertices are elements, 
                    # and we forbid pairs whose sum is in the set.
                    # Actually, sum-free means: no a,b,c in S with a+b=c
                    # This is equivalent to: the set is independent in a graph where
                    # we connect a and b if a+b is in S? No, that's not right either.
                    
                    # Let me think differently: 
                    # A set T is sum-free iff for all a,b in T, a+b not in T.
                    # This is a constraint on triples, not pairs.
                    # 
                    # Better approach: use greedy + local search directly on the set.
                    pass
        
        return adj

    def can_add(x, S_set):
        """Check if x can be added to S_set while maintaining sum-freeness."""
        if 2 * x in S_set:
            return False
        for a in S_set:
            if a + x in S_set:
                return False
            if x - a in S_set:
                return False
        return True

    def greedy_initial(S, order_key):
        """Greedy construction with given ordering."""
        S_sorted = sorted(S, key=order_key)
        result = []
        result_set = set()
        for x in S_sorted:
            if time.time() > DEADLINE - 5:
                break
            if can_add(x, result_set):
                result.append(x)
                result_set.add(x)
        return result

    def simulated_annealing(S, initial_set, time_budget=10.0):
        """SA to escape local optima. Moves: remove one, add one, or swap."""
        current = set(initial_set)
        current_size = len(current)
        best = set(current)
        best_size = current_size
        
        # Temperature schedule
        T0 = 1.0
        T_final = 0.01
        cooling = 0.999
        
        T = T0
        rng = random.Random(42)
        
        while time.time() < t0 + time_budget and T > T_final:
            if time.time() > DEADLINE - 10:
                break
            
            # Generate a move
            move_type = rng.random()
            
            if move_type < 0.3:
                # Remove one element
                if not current:
                    T *= cooling
                    continue
                to_remove = rng.choice(list(current))
                temp_set = set(current)
                temp_set.discard(to_remove)
                delta = -1
            elif move_type < 0.7:
                # Add one element
                candidates = [x for x in S if x not in current]
                if not candidates:
                    T *= cooling
                    continue
                to_add = rng.choice(candidates)
                temp_set = set(current)
                temp_set.add(to_add)
                # Check if valid
                if not is_sum_free(temp_set):
                    T *= cooling
                    continue
                delta = 1
            else:
                # Swap: remove one, add one
                if not current:
                    T *= cooling
                    continue
                to_remove = rng.choice(list(current))
                candidates = [x for x in S if x not in current]
                if not candidates:
                    T *= cooling
                    continue
                to_add = rng.choice(candidates)
                temp_set = set(current)
                temp_set.discard(to_remove)
                temp_set.add(to_add)
                if not is_sum_free(temp_set):
                    T *= cooling
                    continue
                delta = 0  # Same size
            
            # Accept or reject
            if delta >= 0 or rng.random() < (2 ** delta) / T:
                current = temp_set
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(current)
            
            T *= cooling
        
        return list(best)

    def local_search_improve(S, initial_set, max_iters=500):
        """Iterative improvement: try all single additions/removals."""
        current = set(initial_set)
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            if time.time() > DEADLINE - 10:
                break
            improved = False
            iters += 1
            
            # Try adding each candidate
            for x in S:
                if x in current:
                    continue
                if can_add(x, current):
                    current.add(x)
                    improved = True
            
            # Try removing elements that block many additions
            for a in list(current):
                temp = set(current)
                temp.discard(a)
                # Count how many we could add
                addable = 0
                for x in S:
                    if x not in temp and can_add(x, temp):
                        addable += 1
                # If removing a lets us add more than 1, do it
                if addable > 1:
                    current = temp
                    improved = True
        
        return list(current)

    def solve_instance(S, time_budget=30.0):
        """Solve one instance with multiple strategies + SA."""
        start = time.time()
        
        # Generate diverse initial solutions
        strategies = [
            lambda x: x % 2 != 0,  # Odds
            lambda x: (-abs(x), x),  # Large |x| first
            lambda x: (abs(x), x),  # Small |x| first
            lambda x: (-x, abs(x)),  # Positive first
            lambda x: (x, abs(x)),  # Negative first
            lambda x: (random.Random(0).random(), x),  # Random
        ]
        
        best = None
        best_size = 0
        
        for key_func in strategies:
            if time.time() > DEADLINE - 15:
                break
            
            # Greedy initial
            if callable(key_func) and key_func.__code__.co_flags & 0x08:  # Has free vars (lambda with closure)
                # For random, use a different approach
                rng = random.Random(hash(key_func) % 10000)
                S_shuffled = S[:]
                rng.shuffle(S_shuffled)
                initial = []
                init_set = set()
                for x in S_shuffled:
                    if can_add(x, init_set):
                        initial.append(x)
                        init_set.add(x)
            else:
                initial = greedy_initial(S, key_func)
            
            if len(initial) > best_size:
                best = initial
                best_size = len(initial)
            
            # Local search improvement
            if time.time() < DEADLINE - 20:
                improved = local_search_improve(S, initial, max_iters=200)
                if len(improved) > best_size:
                    best = improved
                    best_size = len(improved)
        
        # Simulated annealing on best solution
        if best and time.time() < DEADLINE - 10:
            sa_result = simulated_annealing(S, best, time_budget=min(15.0, DEADLINE - 15 - time.time()))
            if len(sa_result) > best_size:
                best = sa_result
                best_size = len(sa_result)
        
        if best is None:
            best = [x for x in S if x % 2 != 0]
        
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        
        # Allocate time based on instance size
        if size <= 100:
            budget = 25.0
        elif size <= 200:
            budget = 40.0
        else:
            budget = 55.0
        
        best = solve_instance(S, time_budget=budget)
        
        # Final verification
        if not is_sum_free(best):
            best = [x for x in S if x % 2 != 0]
        
        results.append(best)
    
    return results
# EVOLVE-BLOCK-END
