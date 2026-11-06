import time
import random
import sys
from itertools import combinations

N = 256
MAX_EACH = 64
DEADLINE = 55.0

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Number-theoretic construction using modular arithmetic
    # Strategy: Use a prime p and construct Sidon sets via residue classes
    # that guarantee disjoint difference sets
    
    # For a Sidon set in [1, N], we need all pair sums a_i + a_j (i <= j) to be unique
    # For disjoint differences, we need diff(A) ∩ diff(B) = ∅ where diff(S) = {|a-b| : a,b in S, a != b}
    
    # Key insight: If we work modulo a prime p > 2*N, then:
    # - A set S is Sidon mod p iff all (s_i + s_j) mod p are unique for i <= j
    # - We can use quadratic residues and non-residues to construct such sets
    
    # Let's try a constructive approach:
    # 1. Find a prime p where we can fit our sets
    # 2. Construct A and B as scaled versions of known Sidon structures
    
    # Simpler approach: Use the fact that for small enough sets, we can greedily
    # construct with guaranteed properties using modular constraints
    
    # Actually, let's use a direct number-theoretic construction:
    # Take A = {a_i} where a_i are chosen so that differences are from one "range"
    # and B = {b_j} where differences are from another disjoint "range"
    
    # Construction: 
    # Let A be a Sidon set with all differences in [1, D]
    # Let B be a Sidon set with all differences in [D+1, 255]
    # This guarantees disjoint difference sets
    
    # To maximize size, we want both sets as large as possible
    
    # Greedy construction with difference range separation:
    
    def build_sidon_with_diff_range(start_diff, end_diff, elements_pool):
        """Build a Sidon set where all nonzero differences are in [start_diff, end_diff]"""
        result = []
        sums = set()
        diffs = set()
        
        # Sort candidates to try smaller ones first for denser packing
        for x in sorted(elements_pool):
            can_add = True
            
            # Check if adding x keeps all differences in range
            for y in result:
                d = abs(x - y)
                if d < start_diff or d > end_diff:
                    can_add = False
                    break
                if d in diffs:
                    # Difference already used - but that's okay for Sidon, 
                    # we just need pair sums to be unique
                    pass
            
            if not can_add:
                continue
            
            # Check Sidon property: all pair sums must be unique
            if 2 * x in sums:
                continue
            for y in result:
                if x + y in sums:
                    can_add = False
                    break
            
            if not can_add:
                continue
            
            # Add x
            result.append(x)
            sums.add(2 * x)
            for y in result[:-1]:
                sums.add(x + y)
                d = abs(x - y)
                diffs.add(d)
        
        return result
    
    # Try different difference range splits to maximize |A| + |B|
    best_A = []
    best_B = []
    best_total = 0
    
    # The difference range for A should be [1, D] and for B should be [D+1, 255]
    # But we need to be smarter: the actual differences depend on the set elements
    
    # Alternative: Use modular arithmetic with a prime
    # Let p be a prime, and work in Z_p
    # A set S is a Sidon set in Z_p if all (s_i + s_j) mod p are distinct for i <= j
    # This is equivalent to: the equation x + y = z + w has only trivial solutions
    
    # For our purpose, we can use:
    # A = {x in [1,256] : x mod p in R_A}
    # B = {x in [1,256] : x mod p in R_B}
    # where R_A and R_B are chosen so that:
    # 1. R_A is a Sidon set mod p
    # 2. R_B is a Sidon set mod p  
    # 3. (R_A - R_A) mod p and (R_B - R_B) mod p are disjoint (excluding 0)
    
    # Let's try small primes and construct explicitly
    
    # Actually, the simplest guaranteed construction:
    # Use A as a geometric progression-like set with large gaps
    # and B as another set with different gap structure
    
    # Let me try a different approach: constructive with verification
    
    def is_sidon(S):
        """Check if S is a Sidon set (all pair sums unique)"""
        if not S:
            return True
        sums = set()
        for i, a in enumerate(S):
            for j in range(i, len(S)):
                s = a + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def get_nonzero_diffs(S):
        """Get all nonzero absolute differences"""
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = abs(S[i] - S[j])
                if d > 0:
                    diffs.add(d)
        return diffs
    
    def validate(A, B):
        """Validate that A and B are valid Sidon sets with disjoint nonzero differences"""
        if not A or not B:
            return False
        if len(A) > MAX_EACH or len(B) > MAX_EACH:
            return False
        if not is_sidon(A):
            return False
        if not is_sidon(B):
            return False
        diffs_A = get_nonzero_diffs(A)
        diffs_B = get_nonzero_diffs(B)
        if diffs_A & diffs_B:
            return False
        return True
    
    # Construction strategy: 
    # 1. Build A greedily with preference for small elements (dense packing)
    # 2. Build B greedily avoiding A's differences
    
    def greedy_build_A(elements_pool):
        """Greedy build of a Sidon set"""
        A = []
        sums = set()
        diffs = set()
        
        for x in sorted(elements_pool):
            can_add = True
            
            # Check Sidon property
            if 2 * x in sums:
                can_add = False
            else:
                for y in A:
                    if x + y in sums:
                        can_add = False
                        break
            
            if can_add:
                A.append(x)
                sums.add(2 * x)
                for y in A[:-1]:
                    sums.add(x + y)
                    d = abs(x - y)
                    diffs.add(d)
        
        return A, diffs
    
    def greedy_build_B(elements_pool, forbidden_diffs):
        """Greedy build of a Sidon set avoiding forbidden differences"""
        B = []
        sums = set()
        diffs = set()
        
        for x in sorted(elements_pool):
            can_add = True
            
            # Check that no difference with existing B elements is in forbidden_diffs
            for y in B:
                d = abs(x - y)
                if d in forbidden_diffs:
                    can_add = False
                    break
            
            if not can_add:
                continue
            
            # Check Sidon property
            if 2 * x in sums:
                can_add = False
            else:
                for y in B:
                    if x + y in sums:
                        can_add = False
                        break
            
            if can_add:
                B.append(x)
                sums.add(2 * x)
                for y in B[:-1]:
                    sums.add(x + y)
                    d = abs(x - y)
                    diffs.add(d)
        
        return B, diffs
    
    # Try multiple random orderings and difference-range strategies
    rng = random.Random(42)
    
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        
        # Strategy 1: Build A first with small elements, then B avoiding A's diffs
        pool_A = list(range(1, N + 1))
        rng.shuffle(pool_A)
        # Prefer smaller elements for A to allow larger gaps for B
        pool_A.sort(key=lambda x: (x % 7, x))  # Mix small elements with structure
        
        A, diffs_A = greedy_build_A(pool_A)
        
        if not A:
            continue
        
        # Build B avoiding A's differences
        pool_B = list(range(1, N + 1))
        rng.shuffle(pool_B)
        pool_B.sort(key=lambda x: (x % 11, x))
        
        B, diffs_B = greedy_build_B(pool_B, diffs_A)
        
        if validate(A, B):
            total = len(A) + len(B)
            if total > best_total:
                best_total = total
                best_A = list(A)
                best_B = list(B)
        
        # Strategy 2: Alternate building A and B
        if trial % 2 == 1:
            A2 = []
            B2 = []
            sums_A = set()
            sums_B = set()
            diffs_A2 = set()
            diffs_B2 = set()
            
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
            
            for x in candidates:
                # Try to add to A
                can_add_A = True
                if 2 * x in sums_A:
                    can_add_A = False
                else:
                    for y in A2:
                        if x + y in sums_A:
                            can_add_A = False
                            break
                        d = abs(x - y)
                        if d in diffs_A2 or d in diffs_B2:
                            can_add_A = False
                            break
                
                if can_add_A and len(A2) < MAX_EACH:
                    A2.append(x)
                    sums_A.add(2 * x)
                    for y in A2[:-1]:
                        sums_A.add(x + y)
                        diffs_A2.add(abs(x - y))
                    continue
                
                # Try to add to B
                can_add_B = True
                if 2 * x in sums_B:
                    can_add_B = False
                else:
                    for y in B2:
                        if x + y in sums_B:
                            can_add_B = False
                            break
                        d = abs(x - y)
                        if d in diffs_A2 or d in diffs_B2:
                            can_add_B = False
                            break
                
                if can_add_B and len(B2) < MAX_EACH:
                    B2.append(x)
                    sums_B.add(2 * x)
                    for y in B2[:-1]:
                        sums_B.add(x + y)
                        diffs_B2.add(abs(x - y))
            
            if validate(A2, B2):
                total = len(A2) + len(B2)
                if total > best_total:
                    best_total = total
                    best_A = list(A2)
                    best_B = list(B2)
    
    # Fallback: if nothing found, return a minimal valid solution
    if not best_A or not best_B:
        # Minimal valid: A = [1, 3], B = [2, 5]
        # Check: A diffs = {2}, B diffs = {3} - disjoint
        # A sums: 1+1=2, 1+3=4, 3+3=6 - unique
        # B sums: 2+2=4, 2+5=7, 5+5=10 - unique
        # But wait, A and B can share elements, so this is fine
        best_A = [1, 3]
        best_B = [2, 5]
    
    return [best_A, best_B]

if __name__ == "__main__":
    result = construct()
    print(f"A: {len(result[0])} elements, B: {len(result[1])} elements, total: {len(result[0]) + len(result[1])}")
    print(f"A = {result[0]}")
    print(f"B = {result[1]}")
