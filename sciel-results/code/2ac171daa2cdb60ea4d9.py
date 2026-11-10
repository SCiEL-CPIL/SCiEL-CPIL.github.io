"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s limit
    
    # Start with all elements 1..180
    S = set(range(1, 181))
    
    # Precompute for each a, the set of (b,c) pairs that would form a forbidden triple
    # Forbidden triple: distinct a,b,c in S with (b-a)(c-a) = a^2
    # i.e., 1/a = 1/b + 1/c
    
    def get_violations(S):
        """Return dict mapping each element to the set of triples it's in."""
        violations = {}
        S_list = sorted(S)
        for a in S_list:
            a2 = a * a
            # Find all b in S, b != a, such that there exists c in S, c != a, c != b, with (b-a)(c-a) = a2
            # (b-a) must divide a2, and c = a + a2/(b-a)
            bad_b = set()
            for b in S_list:
                if b == a:
                    continue
                diff_b = b - a
                if diff_b == 0:
                    continue
                if a2 % diff_b != 0:
                    continue
                c = a + a2 // diff_b
                if c in S and c != a and c != b:
                    bad_b.add(b)
                    bad_b.add(c)
            if bad_b:
                violations[a] = bad_b
        return violations
    
    # Subtractive greedy: repeatedly remove the element involved in the most violations
    best_set = set(S)
    best_size = len(S)
    
    while time.time() < deadline:
        violations = get_violations(S)
        if not violations:
            break  # No violations, we're done
        
        # Find the element with the most violations (most other elements it conflicts with)
        # We want to remove the element that breaks the most triples
        # Count how many triples each element is part of
        triple_count = {}
        for a, bad_b in violations.items():
            for b in bad_b:
                triple_count[b] = triple_count.get(b, 0) + 1
        
        # Remove the element with highest triple count
        if triple_count:
            worst = max(triple_count, key=lambda x: triple_count[x])
            S.remove(worst)
        else:
            break
        
        if len(S) < best_size:
            # This is getting worse, but we need to remove to eliminate violations
            pass
        best_size = len(S)
    
    # Try to add back elements that were removed
    removed = set(range(1, 181)) - S
    for elem in sorted(removed, reverse=True):
        if time.time() > deadline:
            break
        S_test = S | {elem}
        violations = get_violations(S_test)
        if not violations:
            S = S_test
    
    return sorted(S)

# EVOLVE-BLOCK-END
