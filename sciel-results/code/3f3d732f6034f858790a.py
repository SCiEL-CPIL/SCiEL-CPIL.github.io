"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import itertools

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i, v in enumerate(is_prime) if v]

def construct():
    all_primes = _sieve_primes(20000)
    max_sum = 20000
    max_val = 10000
    
    def get_smooth_mask(primes_set):
        """Returns a boolean list of length 20001 where mask[s] is True if s is P-smooth."""
        mask = [False] * (max_sum + 1)
        for s in range(1, max_sum + 1):
            n = s
            is_smooth = True
            for p in primes_set:
                while n % p == 0:
                    n //= p
                if n == 1:
                    break
            if n > 1:
                is_smooth = False
            mask[s] = is_smooth
        return mask

    def find_clique(mask, size=12):
        """Finds a clique of given size in the graph defined by mask."""
        # Build adjacency lists
        neighbors = [set() for _ in range(max_val + 1)]
        
        # Iterate through smooth sums
        # s = u + v
        # u in [1, 10000], v in [1, 10000], u != v
        for s in range(2, max_sum + 1):
            if not mask[s]:
                continue
            # u ranges from max(1, s - max_val) to min(max_val, s - 1)
            start = max(1, s - max_val)
            end = min(max_val, s - 1)
            for u in range(start, end + 1):
                v = s - u
                if u != v:
                    neighbors[u].add(v)
                    neighbors[v].add(u)
        
        # Filter candidates to those with degree >= size - 1
        candidates = set()
        for u in range(1, max_val + 1):
            if len(neighbors[u]) >= size - 1:
                candidates.add(u)
        
        if not candidates:
            return None
            
        # Greedy with limited backtracking
        def dfs(clique, cand_set):
            if len(clique) == size:
                return clique[:]
            
            if not cand_set:
                return None
            
            # Heuristic: pick candidates with highest degree
            # To speed up, we don't sort the whole set every time if it's large,
            # but we can pick a few high-degree ones.
            
            # Get top 5 candidates by degree
            # Sorting a set of up to 10000 elements is O(N log N) ~ 100k ops.
            # Doing this 12 times is ~1.2M ops. Acceptable.
            
            sorted_cands = sorted(cand_set, key=lambda x: len(neighbors[x]), reverse=True)
            
            # Try top 5
            for c in sorted_cands[:5]:
                new_clique = clique + [c]
                new_cand_set = cand_set & neighbors[c]
                result = dfs(new_clique, new_cand_set)
                if result:
                    return result
            
            return None

        return dfs([], candidates)

    # Try prime sets in increasing order of size
    # Size 1
    for p in all_primes[:10]:
        P = {p}
        mask = get_smooth_mask(P)
        clique = find_clique(mask)
        if clique:
            return clique

    # Size 2
    # Try pairs of small primes
    for i in range(10):
        for j in range(i + 1, 15):
            P = {all_primes[i], all_primes[j]}
            mask = get_smooth_mask(P)
            clique = find_clique(mask)
            if clique:
                return clique

    # Size 3
    # Try combinations of first 5 primes
    for combo in itertools.combinations(all_primes[:5], 3):
        P = set(combo)
        mask = get_smooth_mask(P)
        clique = find_clique(mask)
        if clique:
            return clique

    # Fallback
    return list(range(2, 26, 2))

# EVOLVE-BLOCK-END
