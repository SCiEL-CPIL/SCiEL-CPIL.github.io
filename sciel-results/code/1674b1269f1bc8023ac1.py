import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe
    
    N = 128
    # Pre-compute all potential violation sets
    violations = []
    
    # Generate all triples a < b < c
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            s_ab = a + b
            if s_ab > N:
                continue
            for c in range(b + 1, N + 1):
                s_ac = a + c
                s_bc = b + c
                if s_ac > N or s_bc > N:
                    continue
                viol = (a, b, c, s_ab, s_ac, s_bc)
                violations.append(viol)
    
    # Initial set: all 1..128
    S = [True] * (N + 1)  # 1-indexed
    
    # Helper to check if a violation is active
    def is_active(viol):
        return all(S[x] for x in viol)
    
    # Initial check
    active_violations = [v for v in violations if is_active(v)]
    
    # Greedy removal with smarter heuristic:
    # For each element in active violations, compute a score:
    #   score = count_in_active_violations - lambda * (number of other elements in those violations)
    # This prefers removing elements that break many violations but don't "waste" removals
    # on violations that would be broken by other elements too.
    
    while active_violations:
        if time.time() > deadline:
            break
            
        # Count occurrences and also track which violations each element is in
        counts = {}
        for v in active_violations:
            for x in v:
                if x not in counts:
                    counts[x] = 0
                counts[x] += 1
        
        if not counts:
            break
            
        # Compute a refined score: prefer elements that appear in many violations
        # but penalize elements that are "redundant" (many other elements in same violations)
        # Simple approach: just use count, but break ties by preferring larger elements
        # (larger elements are more likely to be sums, removing them might be more effective)
        
        max_count = max(counts.values())
        candidates = [x for x, c in counts.items() if c == max_count]
        
        # Prefer removing larger elements (they're more likely to be pair sums)
        # among ties, this might help because smaller elements are more "central"
        x_to_remove = max(candidates)
        
        # Remove x_to_remove from S
        S[x_to_remove] = False
        
        # Update active_violations
        new_active = []
        for v in active_violations:
            if x_to_remove not in v:
                new_active.append(v)
        
        active_violations = new_active
    
    # Local search polish: try to add back removed elements if possible
    # An element can be added back if doing so doesn't create any new active violation
    # that isn't already broken by other missing elements
    
    removed_elements = [i for i in range(1, N + 1) if not S[i]]
    
    # Try adding back elements in order of smallest index (less likely to be sums)
    for elem in sorted(removed_elements):
        if time.time() > deadline - 5:
            break
        
        # Check if adding elem would create any active violation
        can_add = True
        for v in violations:
            if elem in v and all(S[x] for x in v if x != elem):
                # This violation would become active if we add elem
                can_add = False
                break
        
        if can_add:
            S[elem] = True
    
    # Second pass: try adding back elements that were skipped
    # This time, we allow adding if the resulting set is still valid
    for elem in sorted(removed_elements, reverse=True):
        if time.time() > deadline - 2:
            break
        
        if S[elem]:
            continue
        
        # Check if adding elem would create any active violation
        can_add = True
        for v in violations:
            if elem in v and all(S[x] for x in v if x != elem):
                can_add = False
                break
        
        if can_add:
            S[elem] = True
    
    # Convert to list of ints
    result = [i for i in range(1, N + 1) if S[i]]
    
    # Ensure types are correct
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
