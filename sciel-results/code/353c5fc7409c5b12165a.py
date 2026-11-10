"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random

def construct():
    N = 1000
    # We want to find a subset S of {1, ..., N} such that every integer has at most
    # two representations a + b with a <= b from S.
    # This is a B_2[2] set.
    
    # Strategy:
    # 1. Start with a random dense initialization.
    # 2. Use local search to remove/swap elements that cause violations.
    # 3. Use NumPy for efficient collision counting via convolution.
    
    # Helper: compute collision counts for a set represented as a boolean array
    def compute_collisions(set_bool):
        # set_bool is a boolean array of length N+1 (indices 0..N)
        # We need to count representations of sums s = a + b with a <= b.
        # Using convolution: conv = set_bool * set_bool
        # But this counts ordered pairs (a,b) and (b,a) separately, and (a,a) once.
        # For a <= b:
        #   If a < b: the pair (a,b) contributes to sum s=a+b. In convolution, 
        #     both (a,b) and (b,a) appear, so count is 2 for each unordered pair with a != b.
        #   If a == b: the pair (a,a) contributes to sum s=2a. In convolution, 
        #     it appears once.
        # 
        # So for a sum s:
        #   Let conv[s] = number of ordered pairs (a,b) with a,b in set, a+b=s.
        #   The number of unordered pairs with a <= b is:
        #     (conv[s] + (1 if s is even and s/2 in set else 0)) // 2
        #   Wait, let's verify:
        #   If a < b: conv[s] includes (a,b) and (b,a), so 2 per unordered pair.
        #   If a == b: conv[s] includes (a,a) once.
        #   So total unordered pairs with a <= b = (conv[s] + I[s even and s/2 in set]) // 2
        #   Actually: conv[s] = 2 * (# pairs with a < b) + (# pairs with a == b)
        #   # pairs with a <= b = (# pairs with a < b) + (# pairs with a == b)
        #   = (conv[s] - (# pairs with a == b)) / 2 + (# pairs with a == b)
        #   = conv[s]/2 + (# pairs with a == b)/2
        #   (# pairs with a == b) is 1 if s is even and s/2 in set, else 0.
        #   So # pairs with a <= b = (conv[s] + I[s even and s/2 in set]) / 2
        
        conv = np.convolve(set_bool, set_bool)
        # conv is a float array, but values are integers
        conv = conv.astype(int)
        
        # For each sum s from 2 to 2*N:
        #   count = (conv[s] + (1 if s % 2 == 0 and s//2 <= N and set_bool[s//2] else 0)) // 2
        #   We need count <= 2
        
        max_count = 0
        for s in range(2, 2 * N + 1):
            if s < len(conv):
                c = conv[s]
                if s % 2 == 0:
                    half = s // 2
                    if 0 <= half <= N and set_bool[half]:
                        c += 1
                # c should be even now? Let's check: 
                # conv[s] = 2*(#a<b) + (#a==b)
                # If s even and s/2 in set, #a==b = 1, so c = 2*(#a<b) + 1 + 1 = 2*(#a<b) + 2 = even
                # If not, #a==b = 0, so c = 2*(#a<b) = even
                # So c is always even.
                count = c // 2
                if count > max_count:
                    max_count = count
        return max_count
    
    # Helper: compute which sums have > 2 representations
    def get_violations(set_bool):
        conv = np.convolve(set_bool, set_bool)
        conv = conv.astype(int)
        violations = set()
        for s in range(2, 2 * N + 1):
            if s < len(conv):
                c = conv[s]
                if s % 2 == 0:
                    half = s // 2
                    if 0 <= half <= N and set_bool[half]:
                        c += 1
                count = c // 2
                if count > 2:
                    violations.add(s)
        return violations
    
    # Start with a random dense set
    # A B_2[2] set has density at most ~2/sqrt(N) per element? 
    # Actually, the asymptotic density is at least 3/2 * sqrt(N) for the size.
    # For N=1000, sqrt(1000) ≈ 31.6, so 3/2 * 31.6 ≈ 47.4
    # But we can do better with local search.
    # Let's start with a random set of size ~200-300 and then optimize.
    
    random.seed(42)
    
    # Initialize with a random set
    target_size = 250
    S = set(random.sample(range(1, N + 1), target_size))
    set_bool = np.zeros(N + 1, dtype=bool)
    for x in S:
        set_bool[x] = True
    
    # Local search: iteratively try to remove or swap elements to reduce violations
    # First, remove elements that cause the most violations
    max_iterations = 500
    for iteration in range(max_iterations):
        violations = get_violations(set_bool)
        if not violations:
            break
        
        # Find elements that are involved in the most violations
        # For each element x in S, count how many violation sums involve x
        # A sum s involves x if there exists y in S such that x + y = s
        # This is expensive to compute directly, so let's just try removing random elements
        
        # Try removing a random element from S that is involved in violations
        # Actually, let's try a smarter approach: for each element, compute its "badness"
        # Badness = number of violation sums s such that there exists y in S with x + y = s
        
        # Simplified: just try removing random elements and see if it helps
        if S:
            # Try removing each element and see which one reduces violations the most
            # This is O(|S| * N) per iteration, which might be too slow
            # Let's just try a few random removals
            for _ in range(10):
                if not S:
                    break
                x = random.choice(list(S))
                # Temporarily remove x
                set_bool[x] = False
                new_violations = get_violations(set_bool)
                if len(new_violations) < len(violations):
                    S.discard(x)
                    violations = new_violations
                else:
                    set_bool[x] = True
    
    # Now try to add elements back
    # Try adding each element from 1..N that is not in S
    added_any = True
    while added_any:
        added_any = False
        # Try adding elements in a random order
        candidates = [x for x in range(1, N + 1) if x not in S]
        random.shuffle(candidates)
        for x in candidates:
            # Try adding x
            set_bool[x] = True
            violations = get_violations(set_bool)
            if not violations:
                S.add(x)
                added_any = True
            else:
                set_bool[x] = False
    
    # Try more local search: swap elements
    # Try swapping an element in S with one not in S
    for _ in range(100):
        if not S:
            break
        x = random.choice(list(S))
        y = random.choice([z for z in range(1, N + 1) if z not in S])
        
        # Try swap
        set_bool[x] = False
        set_bool[y] = True
        violations = get_violations(set_bool)
        if not violations:
            S.discard(x)
            S.add(y)
        else:
            set_bool[y] = False
            set_bool[x] = True
    
    return sorted(S)

# EVOLVE-BLOCK-END
