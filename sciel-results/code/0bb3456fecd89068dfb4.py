import random
import time

def construct():
    N = 256
    deadline = time.time() + 85  # Leave 5s margin
    
    # Precompute pairs for each sum m
    # pairs_for_sum[m] = list of (b, c) such that b <= c and b + c = m
    pairs_for_sum = [[] for _ in range(2 * N + 1)]
    for m in range(2, 2 * N + 1):
        # b ranges from 1 to m//2
        # c = m - b
        # We only care about b, c in [1, N]
        # Also b <= c implies b <= m/2
        for b in range(1, m // 2 + 1):
            c = m - b
            if c <= N:
                pairs_for_sum[m].append((b, c))
    
    # Precompute multiples for each a
    # multiples[a] = list of m such that m is a multiple of a and m > 2a (since b>a, c>=b => b+c > 2a)
    # Actually b+c can be 2a if b=a, c=a, but b>a so b+c >= (a+1)+(a+1) = 2a+2.
    # So m must be at least 2a+2.
    # Max m is 2N.
    multiples = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        m = 2 * a
        while m <= 2 * N:
            # We need b > a.
            # If m < 2*a + 2, no b > a can satisfy b+c=m with c>=b.
            # Min b is a+1, min c is a+1, sum 2a+2.
            if m >= 2 * a + 2:
                multiples[a].append(m)
            m += a
    
    def is_valid(S_set):
        """Check if S_set is valid. S_set is a set of ints."""
        for a in S_set:
            for m in multiples[a]:
                # Check if there exists (b, c) in pairs_for_sum[m] such that b > a and b in S_set and c in S_set
                # Note: b <= c is guaranteed by construction.
                # We must ensure b != a and c != a.
                # Since b > a, b != a.
                # If c == a, then b + a = m => b = m - a.
                # If b > a and c = a, then b > c, which contradicts b <= c.
                # So if b > a, then c >= b > a. So c != a is automatic.
                for b, c in pairs_for_sum[m]:
                    if b > a:
                        if b in S_set and c in S_set:
                            return False
        return True
    
    def get_violations(S_set):
        """Return a list of violating triples (a, b, c)."""
        violations = []
        for a in S_set:
            for m in multiples[a]:
                for b, c in pairs_for_sum[m]:
                    if b > a:
                        if b in S_set and c in S_set:
                            violations.append((a, b, c))
        return violations
    
    def repair(S_set):
        """Remove elements to make S_set valid. Returns the repaired set."""
        S_set = set(S_set)
        while True:
            if time.time() > deadline:
                break
            violations = get_violations(S_set)
            if not violations:
                break
            # Count frequency of each element in violations
            counts = {}
            for a, b, c in violations:
                counts[a] = counts.get(a, 0) + 1
                counts[b] = counts.get(b, 0) + 1
                counts[c] = counts.get(c, 0) + 1
            # Remove the element with the highest count
            # If tie, pick randomly
            max_count = max(counts.values())
            candidates = [x for x, cnt in counts.items() if cnt == max_count]
            to_remove = random.choice(candidates)
            S_set.remove(to_remove)
        return S_set
    
    def improve(S_set):
        """Try to add elements back to S_set."""
        S_set = set(S_set)
        remaining = set(range(1, N + 1)) - S_set
        # Shuffle remaining to try different orders
        remaining_list = list(remaining)
        random.shuffle(remaining_list)
        
        improved = False
        for x in remaining_list:
            if time.time() > deadline:
                break
            # Try adding x
            if is_valid(S_set | {x}):
                S_set.add(x)
                improved = True
        return S_set
    
    best_set = set()
    
    # Run multiple restarts
    while time.time() < deadline:
        # Start with full set or a random subset?
        # Full set requires more repairs.
        # Let's try starting with a random subset of size ~150-180
        # Or just full set.
        initial_set = set(range(1, N + 1))
        
        # Randomize the repair process by shuffling the set?
        # The repair function picks randomly from max count candidates, so it's already randomized.
        
        repaired = repair(initial_set)
        improved = improve(repaired)
        
        if len(improved) > len(best_set):
            best_set = improved
            
    return list(best_set)
