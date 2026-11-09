"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import random

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def greedy_extend_sum_free(base_set, candidates):
        """
        Greedily try to add candidates to base_set while maintaining sum-free property.
        base_set: list of elements already in the sum-free subset.
        candidates: list of elements to try adding.
        """
        current_set = set(base_set)
        result = list(base_set)
        
        # Sort candidates by absolute value descending
        for cand in sorted(candidates, key=lambda x: -abs(x)):
            if cand in current_set:
                continue
            
            valid = True
            
            # Check 1: Is there a, b in current_set such that a + b == cand?
            for a in current_set:
                if (cand - a) in current_set:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check 2: Does cand combine with any existing element to form another existing element?
            # i.e., exists x in current_set such that cand + x in current_set?
            for x in current_set:
                if (x + cand) in current_set:
                    valid = False
                    break
            
            if valid:
                current_set.add(cand)
                result.append(cand)
                
        return result

    def solve_sum_free(elements):
        if not elements:
            return []
        
        M = max(abs(x) for x in elements)
        
        # Initial candidate: elements with magnitude > M/2
        # This set is guaranteed sum-free.
        base = [x for x in elements if abs(x) > M / 2.0]
        
        # Candidates to add: the rest
        rest = [x for x in elements if abs(x) <= M / 2.0]
        
        # Try to extend
        result = greedy_extend_sum_free(base, rest)
        
        # Alternative strategy: Pure greedy from scratch sorted by abs value descending
        greedy_result = []
        greedy_set = set()
        for x in sorted(elements, key=lambda x: -abs(x)):
            can_add = True
            for a in greedy_set:
                if (x - a) in greedy_set:
                    can_add = False
                    break
            if can_add:
                for a in greedy_set:
                    if (a + x) in greedy_set:
                        can_add = False
                        break
            if can_add:
                greedy_result.append(x)
                greedy_set.add(x)
        
        if len(greedy_result) > len(result):
            result = greedy_result
            
        return result

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elems = instance(size, seed)
        subset = solve_sum_free(elems)
        results.append(subset)

    return results
# EVOLVE-BLOCK-END
