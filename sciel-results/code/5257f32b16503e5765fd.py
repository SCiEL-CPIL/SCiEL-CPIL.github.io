"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations_with_replacement, combinations

def construct():
        n = 256
        best_total = 0
        best_A = None
        best_B = None
        
        # Increase seeds for better search coverage
        for seed in range(1000):
            rng = random.Random(seed)
            A = []
            B = []
            sums_A = set()
            sums_B = set()
            diffs_A = set()
            diffs_B = set()
            
            # Shuffle the order of elements
            elements = list(range(1, n+1))
            rng.shuffle(elements)
            unused = set(elements)
            
            def try_add_A(x):
                nonlocal A, sums_A, diffs_A, diffs_B
                if x + x in sums_A:
                    return False
                for s in A:
                    if x + s in sums_A:
                        return False
                    d1 = x - s
                    d2 = s - x
                    if d1 in diffs_B or d2 in diffs_B:
                        return False
                return True

            def try_add_B(x):
                nonlocal B, sums_B, diffs_B, diffs_A
                if x + x in sums_B:
                    return False
                for s in B:
                    if x + s in sums_B:
                        return False
                    d1 = x - s
                    d2 = s - x
                    if d1 in diffs_A or d2 in diffs_A:
                        return False
                return True

            def add_to_A(x):
                nonlocal A, sums_A, diffs_A
                A.append(x)
                sums_A.add(x + x)
                for s in A[:-1]:
                    sums_A.add(x + s)
                    diffs_A.add(x - s)
                    diffs_A.add(s - x)

            def add_to_B(x):
                nonlocal B, sums_B, diffs_B
                B.append(x)
                sums_B.add(x + x)
                for s in B[:-1]:
                    sums_B.add(x + s)
                    diffs_B.add(x - s)
                    diffs_B.add(s - x)

            # Phase 1: Randomized Greedy with Balancing
            for x in elements:
                can_A = try_add_A(x)
                can_B = try_add_B(x)
                
                if can_A and can_B:
                    # Balance: add to smaller set
                    if len(A) <= len(B):
                        add_to_A(x)
                    else:
                        add_to_B(x)
                elif can_A:
                    add_to_A(x)
                elif can_B:
                    add_to_B(x)
                else:
                    # Leave in unused implicitly, as we are iterating over a fixed list
                    # We need to track unused for phase 2
                    pass
            
            # Reconstruct unused set
            used = set(A) | set(B)
            unused = set(elements) - used
            
            # Phase 2: Repair Pass (Greedy on remaining)
            # Try to add remaining elements in sorted order to fill gaps
            # We do multiple passes until no improvement
            improved = True
            while improved:
                improved = False
                # Sort unused to ensure deterministic tie-breaking in repair
                for x in sorted(list(unused)):
                    if try_add_A(x):
                        add_to_A(x)
                        unused.remove(x)
                        improved = True
                        break
                    if try_add_B(x):
                        add_to_B(x)
                        unused.remove(x)
                        improved = True
                        break
            
            total = len(A) + len(B)
            if total > best_total:
                best_total = total
                best_A = A[:]
                best_B = B[:]
        
        return [best_A, best_B]
    # EVOLVE-BLOCK-END
