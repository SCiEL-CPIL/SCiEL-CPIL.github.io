"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 60.0  # Safety margin for 90s limit
    
    # Base set: 64 to 128
    S = set(range(64, 129))
    
    # Candidate small numbers to add
    # We want to find the largest subset X of {1, ..., 63}
    # such that S U X is valid.
    
    # Condition for validity of S U X:
    # No distinct a, b, c in S U X such that a+b, a+c, b+c are all in S U X.
    
    # Since sums of two elements in S are > 128, they are never in S U X.
    # So any triple containing two or more elements from S is safe.
    # We only need to check triples with 1 or 2 elements from X.
    
    # Case 1: Triple is {x, y, z} with x, y, z in X.
    # Need: NOT (x+y in S U X AND x+z in S U X AND y+z in S U X)
    
    # Case 2: Triple is {x, y, a} with x, y in X, a in S.
    # Sums: x+y, x+a, y+a.
    # x+a is in S U X if x+a >= 64 and x+a <= 128.
    # y+a is in S U X if y+a >= 64 and y+a <= 128.
    # x+y is in S U X if x+y in X or (x+y >= 64 and x+y <= 128).
    
    # Case 3: Triple is {x, a, b} with x in X, a, b in S.
    # Sums: x+a, x+b, a+b.
    # a+b > 128, so a+b not in S U X. Safe.
    
    # So we only need to enforce:
    # 1. For all distinct x,y,z in X: not (x+y in S_U_X and x+z in S_U_X and y+z in S_U_X)
    # 2. For all distinct x,y in X and a in S: not (x+y in S_U_X and x+a in S_U_X and y+a in S_U_X)
    
    # Note: If x+y < 64, it's in S_U_X iff x+y in X.
    # If x+y >= 64, it's in S_U_X iff x+y <= 128.
    
    X = []
    # Try adding numbers from 1 to 63
    # Greedy approach: try to add each number if it maintains validity.
    # To get a better set, we might need backtracking or a smarter order.
    # However, a greedy approach on small numbers often works well for this structure.
    # Let's try adding in increasing order.
    
    def is_valid_candidate(x, current_X):
        # Check Case 1: Triples within X U {x}
        # We need to check if adding x creates a violation with existing pairs in X.
        # Violation if there exist distinct y, z in current_X such that:
        # (x+y in S_U_X_new) AND (x+z in S_U_X_new) AND (y+z in S_U_X_new)
        # Note: S_U_X_new = S U current_X U {x}
        
        # Check Case 2: Triples with one new x and one old y in X, and a in S.
        # Violation if there exists y in current_X and a in S such that:
        # (x+y in S_U_X_new) AND (x+a in S_U_X_new) AND (y+a in S_U_X_new)
        
        # Optimization: 
        # If x+y < 64, x+y in S_U_X_new iff x+y in current_X or x+y == x (impossible for distinct) or x+y == y (impossible). 
        # Wait, if x+y is in current_X, it's in S_U_X_new.
        
        # Let's implement the checks directly.
        
        # Precompute sums of pairs in current_X to check against S and current_X
        # But current_X changes.
        
        # For efficiency, we can just iterate. Size of X is small (< 30).
        
        # Check Case 1
        for i in range(len(current_X)):
            y = current_X[i]
            for j in range(i+1, len(current_X)):
                z = current_X[j]
                if x == y or x == z or y == z: continue # distinctness handled by loop structure mostly, but x is new
                
                sum_xy = x + y
                sum_xz = x + z
                sum_yz = y + z
                
                # Check if sums are in S U current_X U {x}
                # A number v is in S_U_X_new if:
                # v in S (64 <= v <= 128) OR v in current_X OR v == x
                
                def in_new_set(v):
                    if 64 <= v <= 128: return True
                    if v in current_X: return True
                    if v == x: return True
                    return False
                
                if in_new_set(sum_xy) and in_new_set(sum_xz) and in_new_set(sum_yz):
                    return False
        
        # Check Case 2
        # x in X_new, y in current_X, a in S
        # Sums: x+y, x+a, y+a
        # a ranges from 64 to 128.
        # x+a is in S_U_X_new if 64 <= x+a <= 128.
        # y+a is in S_U_X_new if 64 <= y+a <= 128.
        # x+y is in S_U_X_new if (64 <= x+y <= 128) or (x+y in current_X) or (x+y == x, impossible).
        
        for y in current_X:
            sum_xy = x + y
            xy_in = (64 <= sum_xy <= 128) or (sum_xy in current_X)
            # Note: sum_xy cannot be x since y>0
            
            if not xy_in:
                continue
                
            # If xy_in is True, we need to check if there exists a in S 
            # such that x+a in S_U_X_new and y+a in S_U_X_new.
            # x+a in S_U_X_new iff 64 <= x+a <= 128.
            # y+a in S_U_X_new iff 64 <= y+a <= 128.
            
            # Let A = {a in S | 64 <= x+a <= 128}
            # Let B = {a in S | 64 <= y+a <= 128}
            # If A and B are not disjoint, there is a violation.
            
            # x+a >= 64 => a >= 64-x
            # x+a <= 128 => a <= 128-x
            # So A is [max(64, 64-x), min(128, 128-x)] intersect S.
            # Since x >= 1, 64-x < 64. So lower bound is 64.
            # Upper bound is 128-x.
            # So A = [64, 128-x] if 128-x >= 64 => x <= 64.
            
            # Similarly B = [64, 128-y] if y <= 64.
            
            # Intersection of [64, 128-x] and [64, 128-y] is [64, min(128-x, 128-y)].
            # This intersection is non-empty if min(128-x, 128-y) >= 64.
            # i.e., 128 - max(x,y) >= 64 => max(x,y) <= 64.
            # Since x,y < 64, this is always true.
            
            # So if xy_in is True, there is ALWAYS a violation?
            # Let's re-read carefully.
            # Violation if there exists a in S such that x+a in S_U_X_new AND y+a in S_U_X_new.
            # We established that if x,y < 64, then for any a in [64, 128-max(x,y)], both x+a and y+a are in S (and thus in S_U_X_new).
            # So if x+y is in S_U_X_new, we have a violation.
            
            # Therefore, for any distinct x, y in X, we MUST have x+y NOT in S_U_X_new.
            # Since x,y < 64, x+y < 128.
            # x+y in S_U_X_new iff (x+y in S) or (x+y in X).
            # x+y in S iff x+y >= 64.
            # So if x+y >= 64, it is in S, so it is in S_U_X_new. Violation!
            # Thus, we MUST have x+y < 64 for all distinct x, y in X.
            
            # This is a strong constraint: X must be a subset of {1, ..., 63} such that the sum of any two distinct elements is < 64.
            # This implies that if we sort X, x_i + x_{i+1} < 64.
            
            # Wait, is this correct?
            # If x+y < 64, then x+y is not in S.
            # It is in S_U_X_new only if x+y in X.
            # So if x+y < 64 and x+y in X, then xy_in is True.
            # Then we need to check if there is an 'a' causing violation.
            # As shown above, if xy_in is True, there IS a violation.
            # So we must avoid xy_in being True.
            # So for all distinct x,y in X, x+y must NOT be in S_U_X_new.
            # This means:
            # 1. If x+y >= 64, then x+y is in S, so it's in S_U_X_new. Violation.
            #    So we MUST have x+y < 64 for all distinct x,y in X.
            # 2. If x+y < 64, then x+y is in S_U_X_new iff x+y in X.
            #    So we MUST have x+y NOT in X for all distinct x,y in X.
            
            # So X must be a sum-free set (distinct sums not in X) AND max pair sum < 64.
            
        return True

    # Greedy construction of X
    # We want to maximize |X|.
    # Constraint: For all distinct x,y in X, x+y < 64 and x+y not in X.
    
    # Let's try to build X.
    # If we pick small numbers, we can pick more.
    # e.g., 1, 2, ...
    # 1+2=3 (in X? no), 1+3=4 (in X? no)...
    # If we pick 1..k, sum of 1 and 2 is 3. If 3 is in X, violation.
    # So X cannot contain 1, 2, 3.
    # It's a "weakly sum-free" set with max sum < 64.
    
    # Let's use a simple greedy: iterate 1..63, add if valid.
    # But order matters. Smaller numbers constrain larger sums less?
    # Actually, smaller numbers allow more room.
    # Let's try adding in increasing order.
    
    current_X = []
    for candidate in range(1, 64):
        if time.time() > deadline - 5:
            break
        if is_valid_candidate(candidate, current_X):
            current_X.append(candidate)
    
    # Combine
    result = list(S | set(current_X))
    result.sort()
    return result

# EVOLVE-BLOCK-END
