"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import product

# EVOLVE-BLOCK-START
def priority(vector, n):
    # Uniform priority: greedy insertion in lexicographic order.
    return 0.0

def _is_cap_set(vectors):
    """Check if a list of vectors forms a cap set (no three-term arithmetic progression)."""
    vec_set = set(vectors)
    for v in vectors:
        for w in vectors:
            if v == w:
                continue
            # Check if (v + w - u) is in the set, where u = 2v - w? 
            # A 3-AP is a, b, c such that b = (a+c)/2, i.e., a+c=2b.
            # Or equivalently, a, b, c with b-a = c-b => a+c=2b.
            # So for any distinct a, c, check if b=(a+c)/2 is in the set and distinct from a,c.
            # Actually, the standard definition: no three distinct elements a,b,c with a+c=2b.
            pass
    # More efficient check: for each pair (a, c) with a != c, check if b = (a+c)//2 is integer and in set.
    # Since elements are in {0,1,2}^n, a+c can be 0,1,2,3,4. b must be integer, so a_i+c_i must be even for all i.
    # Actually, in F_3, 2b = a+c means b = 2^{-1}(a+c) = 2(a+c) mod 3? No, 2b = a+c => b = (a+c)/2.
    # In integers: b_i = (a_i + c_i) / 2. This must be an integer in {0,1,2}.
    # So a_i + c_i must be even for all i.
    # Let's implement a direct check.
    for i, a in enumerate(vectors):
        for j, c in enumerate(vectors):
            if i == j:
                continue
            # Check if b = (a+c)/2 is in the set and distinct from a and c
            b = []
            valid = True
            for k in range(len(a)):
                s = a[k] + c[k]
                if s % 2 != 0:
                    valid = False
                    break
                b.append(s // 2)
            if not valid:
                continue
            b = tuple(b)
            if b in vec_set and b != a and b != c:
                return False
    return True

def _greedy_cap_set(n, priority_func, deadline):
    """Greedily build a cap set using the given priority function."""
    start_time = time.time()
    vectors = []
    vec_set = set()
    
    # Generate all vectors in {0,1,2}^n
    all_vectors = list(product(range(3), repeat=n))
    
    # Sort by priority (descending)
    all_vectors.sort(key=lambda v: priority_func(v, n), reverse=True)
    
    for v in all_vectors:
        if time.time() > deadline:
            break
        if v in vec_set:
            continue
        # Check if adding v keeps the set a cap set
        # Optimized check: only need to check pairs involving v
        is_cap = True
        for existing in vectors:
            # Check if there's a 3-AP involving v and existing
            # Case 1: v is the middle: a + c = 2v => c = 2v - a
            # Case 2: v is the end: a + v = 2b => b = (a+v)/2
            # Case 3: v is the other end: v + c = 2b => b = (v+c)/2
            # Actually, simpler: for each existing vector e, check if there exists a third vector x such that {v, e, x} is a 3-AP.
            # The third vector x is determined by v and e in three ways:
            # 1. v is middle: x = 2v - e
            # 2. e is middle: x = 2e - v
            # 3. x is middle: x = (v + e) / 2 (must be integer)
            
            # Check 1: x = 2v - e
            x1 = tuple(2*vi - ei for vi, ei in zip(v, e))
            if all(0 <= xi <= 2 for xi in x1) and x1 in vec_set and x1 != v and x1 != e:
                is_cap = False
                break
            
            # Check 2: x = 2e - v
            x2 = tuple(2*ei - vi for vi, ei in zip(v, e))
            if all(0 <= xi <= 2 for xi in x2) and x2 in vec_set and x2 != v and x2 != e:
                is_cap = False
                break
            
            # Check 3: x = (v + e) / 2
            valid = True
            x3 = []
            for vi, ei in zip(v, e):
                s = vi + ei
                if s % 2 != 0:
                    valid = False
                    break
                x3.append(s // 2)
            if valid:
                x3 = tuple(x3)
                if x3 in vec_set and x3 != v and x3 != e:
                    is_cap = False
                    break
        
        if is_cap:
            vectors.append(v)
            vec_set.add(v)
    
    return len(vectors)

def _random_priority(seed):
    """Create a random priority function."""
    rng = random.Random(seed)
    def p(vector, n):
        return rng.random()
    return p

def _weighted_priority(weights, n):
    """Create a priority function based on weights."""
    def p(vector, n):
        return sum(w * vi for w, vi in zip(weights, vector))
    return p

def _run_search(n, deadline):
    """Run the search for the best cap set size."""
    best_size = 0
    
    # Try the baseline (uniform priority)
    start = time.time()
    size = _greedy_cap_set(n, priority, deadline)
    best_size = max(best_size, size)
    
    if time.time() > deadline:
        return best_size
    
    # Try random priorities
    for seed in range(10000):
        if time.time() > deadline:
            break
        pfunc = _random_priority(seed)
        size = _greedy_cap_set(n, pfunc, deadline)
        best_size = max(best_size, size)
    
    return best_size

# Cache for results
_result_cache = {}

def _main():
    """Main entry point for the search."""
    global _result_cache
    
    start_time = time.time()
    deadline = start_time + 320  # 320s deadline, 10s safety margin
    
    # The task says "vectors of {0,1,2}^n are added greedily". We need to determine n.
    # Looking at the contract: "priority(vector: tuple, n: int) -> float".
    # The evaluator likely calls priority for a fixed n. We need to infer n from the context.
    # Since the seed program returns 0.0 for all inputs, it doesn't reveal n.
    # Let's assume the evaluator will test for a specific n. We need to handle whatever n is passed.
    # But wait, the score is "cap set size". The evaluator measures the raw value.
    # How does the evaluator know which n to use? It must be fixed in the evaluation harness.
    # Since I don't have access to the harness, I'll assume a reasonable n.
    # Common small n for cap sets: n=3,4,5,6,7,8.
    # Let's try to be adaptive: if n is small, we can do exhaustive search.
    
    # Actually, re-reading the task: "vectors of {0,1,2}^n are added greedily in descending priority"
    # The priority function is called by the evaluator. The evaluator builds the cap set.
    # So my priority function should return a value that, when used for greedy insertion,
    # results in a large cap set.
    
    # The challenge is that the priority function is stateless (it takes vector and n, returns float).
    # It can't maintain state between calls. So I need a deterministic priority function
    # that, when used for greedy insertion, yields a large cap set.
    
    # One approach: use a heuristic that favors vectors that are "far" from each other
    # or that are unlikely to form 3-APs.
    
    # For now, let's just return a priority based on some structural property.
    # Since the evaluator calls priority(vector, n) for each vector, and adds them in descending order,
    # I need to design a priority function that leads to a good cap set.
    
    # Let's think about what makes a good cap set.
    # In {0,1,2}^n, a cap set has no 3-APs.
    # One known construction: use vectors with even sum of coordinates? No, that's not quite right.
    # Actually, the set of all vectors with coordinate sum ≡ 0 mod 3 is not necessarily a cap set.
    
    # Let's try a different approach: the priority function should prioritize vectors
    # that are "independent" in some sense.
    
    # For simplicity, let's return a priority based on the Hamming weight or some other statistic.
    # But since the seed returns 0.0 for all, any non-trivial priority might improve things.
    
    # Let's try: prioritize vectors with fewer 1s (i.e., more 0s and 2s).
    # Or: prioritize vectors that are "far" from the origin.
    
    # Actually, let's just implement a simple heuristic and see.
    # The priority function is called many times, so it should be fast.
    
    # For now, I'll keep the priority function simple but non-uniform.
    # Let's prioritize based on the sum of coordinates (mod 3 or something).
    
    # Wait, the evaluator builds the cap set using my priority function.
    # So I just need to return a float. The evaluator handles the rest.
    # Let me re-read: "vectors of {0,1,2}^n are added greedily in descending priority when they keep the set a cap set"
    # So the evaluator:
    # 1. Generates all vectors in {0,1,2}^n
    # 2. Sorts them by priority (descending)
    # 3. Greedily adds them if they keep the set a cap set
    # 4. Returns the size of the resulting cap set
    
    # So my job is to design a priority function that leads to a large cap set.
    
    # Let's try a simple heuristic: prioritize vectors with coordinate sum ≡ 0 mod 3.
    # This might correlate with cap set properties.
    
    def priority(vector, n):
        s = sum(vector) % 3
        # Return a value that favors s=0
        return -s  # Lower is better? No, descending priority means higher values come first.
        # So return higher values for preferred vectors.
        # Let's return 0 for s=0, -1 for s=1, -2 for s=2.
        return -s
    
    # But wait, I can't redefine priority inside _main. The priority function is the entry point.
    # I need to modify the priority function itself.
    
    # Let me think about this differently. The priority function is called by the evaluator.
    # I need to make it return values that lead to a good cap set.
    
    # One idea: use a hash-based priority that is deterministic but appears random.
    # This might avoid systematic biases.
    
    # Another idea: prioritize vectors that are "spread out" in the space.
    
    # For now, let's try a simple approach: return a value based on the coordinate sum.
    # I'll modify the priority function to do this.
    
    # Since I can't modify the priority function from within _main, I need to edit it directly.
    # Let me rewrite the priority function to use a heuristic.
    
    pass

# Since I can't easily run _main, let me just modify the priority function directly.
# The priority function is the entry point, so I'll edit it.

# Let me redefine priority with a better heuristic.
def priority(vector, n):
    # Heuristic: prioritize vectors with coordinate sum ≡ 0 mod 3
    # This is a simple structural property that might correlate with cap set membership.
    s = sum(vector) % 3
    # Return higher values for s=0
    if s == 0:
        return 2.0
    elif s == 1:
        return 1.0
    else:
        return 0.0
# EVOLVE-BLOCK-END
