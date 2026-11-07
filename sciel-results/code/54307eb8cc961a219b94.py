"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_latin(sq, n):
    for i in range(n):
        if sorted(sq[i]) != list(range(n)):
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True

def _are_orthogonal(sq1, sq2, n):
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (sq1[i][j], sq2[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return True

def _pair_to_int(a, b):
    """Convert (a in Z_2, b in Z_5) to integer 0..9"""
    return a * 5 + b

def _int_to_pair(x):
    """Convert integer 0..9 to (a in Z_2, b in Z_5)"""
    return (x // 5, x % 5)

def construct():
    n = 10
    start = time.time()
    deadline = start + 50.0  # safety margin before 60s limit
    
    # Generate candidate Latin squares using linear forms over Z_2 x Z_5
    # General form: 
    #   a_out = (c2_1 * i2 + c2_2 * j2 + d2) % 2
    #   b_out = (c5_1 * i5 + c5_2 * j5 + d5) % 5
    # For Latin property, we need the maps to be bijective in each row and column.
    
    candidates = []
    
    # Try all combinations of coefficients
    for c2_1 in range(2):
        for c2_2 in range(2):
            for c5_1 in range(5):
                for c5_2 in range(5):
                    # Check if this gives a Latin square
                    # Row condition: for fixed i, as j varies 0..9, 
                    # (j2, j5) takes all 10 pairs. 
                    # a_out = (c2_1*i2 + c2_2*j2 + d2) % 2
                    # b_out = (c5_1*i5 + c5_2*j5 + d5) % 5
                    # For fixed i, i2 and i5 are fixed.
                    # As j varies, j2 in {0,1} (each 5 times), j5 in {0,1,2,3,4} (each 2 times)
                    # We need (a_out, b_out) to cover all 10 pairs.
                    # a_out depends only on j2: takes values (c2_1*i2 + c2_2*0 + d2)%2 and (c2_1*i2 + c2_2*1 + d2)%2
                    #   If c2_2 == 0, a_out is constant -> not Latin unless combined with b_out
                    #   If c2_2 == 1, a_out takes both values
                    # b_out = (c5_1*i5 + c5_2*j5 + d5) % 5
                    #   As j5 varies 0..4, b_out = (c5_1*i5 + c5_2*j5 + d5) % 5
                    #   If c5_2 != 0, this covers all of Z_5 as j5 varies
                    #   If c5_2 == 0, b_out is constant -> not Latin
                    
                    # For the row to be Latin, we need:
                    # - The map j -> (a_out, b_out) to be bijective
                    # j = (j2, j5) where j2 = j//5, j5 = j%5
                    # a_out = (c2_1*i2 + c2_2*j2 + d2) % 2
                    # b_out = (c5_1*i5 + c5_2*j5 + d5) % 5
                    
                    # If c2_2 != 0 and c5_2 != 0:
                    #   For fixed i, a_out takes 2 values (one for j2=0, one for j2=1)
                    #   b_out takes 5 values as j5 varies (if c5_2 != 0)
                    #   But we need 10 distinct pairs from 10 positions
                    #   For j2=0: 5 positions with j5=0..4, a_out = A0, b_out = (c5_1*i5 + c5_2*j5 + d5)%5 -> 5 distinct b values
                    #   For j2=1: 5 positions with j5=0..4, a_out = A1, b_out = (c5_1*i5 + c5_2*j5 + d5)%5 -> same 5 distinct b values
                    #   So we get (A0, b) for 5 values and (A1, b) for 5 values = 10 distinct pairs if A0 != A1
                    #   A0 != A1 iff c2_2 != 0 (since A1 - A0 = c2_2 mod 2)
                    
                    # Similarly for columns.
                    
                    # So conditions for Latin: c2_2 != 0 AND c5_2 != 0
                    # Wait, that's not symmetric. Let me check columns.
                    # Column j fixed: j2, j5 fixed. As i varies 0..9:
                    # i2 = i//5, i5 = i%5
                    # a_out = (c2_1*i2 + c2_2*j2 + d2) % 2
                    # b_out = (c5_1*i5 + c5_2*j5 + d5) % 5
                    # For fixed j, as i varies:
                    # i2 takes values 0 (5 times) and 1 (5 times)
                    # i5 takes values 0,1,2,3,4 (each 2 times)
                    # a_out = (c2_1*i2 + const) % 2
                    #   If c2_1 != 0, takes both values
                    #   If c2_1 == 0, constant
                    # b_out = (c5_1*i5 + const) % 5
                    #   If c5_1 != 0, takes all 5 values as i5 varies
                    #   If c5_1 == 0, constant
                    
                    # For column to be Latin: c2_1 != 0 AND c5_1 != 0
                    
                    # So for both row and column Latin: c2_1 != 0, c2_2 != 0, c5_1 != 0, c5_2 != 0
                    # But c2_1 and c2_2 are in Z_2, so c2_1=1, c2_2=1
                    # c5_1 and c5_2 are in Z_5, so c5_1 != 0, c5_2 != 0
                    
                    if c2_1 == 0 or c2_2 == 0 or c5_1 == 0 or c5_2 == 0:
                        continue
                    
                    # Generate the square for d2 in 0..1, d5 in 0..4
                    for d2 in range(2):
                        for d5 in range(5):
                            sq = []
                            for i in range(10):
                                i2, i5 = i // 5, i % 5
                                row = []
                                for j in range(10):
                                    j2, j5 = j // 5, j % 5
                                    a_out = (c2_1 * i2 + c2_2 * j2 + d2) % 2
                                    b_out = (c5_1 * i5 + c5_2 * j5 + d5) % 5
                                    row.append(_pair_to_int(a_out, b_out))
                                sq.append(row)
                            if _is_latin(sq, 10):
                                candidates.append(sq)
    
    # Also include the original simpler family
    for c2 in range(2):
        for c5 in range(5):
            sq = []
            for i in range(10):
                i2, i5 = i // 5, i % 5
                row = []
                for j in range(10):
                    j2, j5 = j // 5, j % 5
                    a_out = (c2 * i2 + j2) % 2
                    b_out = (c5 * i5 + j5) % 5
                    row.append(_pair_to_int(a_out, b_out))
                sq.append(row)
            if _is_latin(sq, 10):
                # Check if already in candidates
                found = False
                for existing in candidates:
                    if existing == sq:
                        found = True
                        break
                if not found:
                    candidates.append(sq)
    
    # Use backtracking to find maximum set of mutually orthogonal squares
    best_set = []
    
    def backtrack(remaining, current_set):
        nonlocal best_set
        if time.time() > deadline:
            return
        if len(current_set) > len(best_set):
            best_set = list(current_set)
        if not remaining:
            return
        for idx, sq in enumerate(remaining):
            ok = True
            for sq2 in current_set:
                if not _are_orthogonal(sq, sq2, 10):
                    ok = False
                    break
            if ok:
                current_set.append(sq)
                backtrack(remaining[:idx] + remaining[idx+1:], current_set)
                current_set.pop()
    
    backtrack(candidates, [])
    
    if len(best_set) >= 1:
        result = best_set
    else:
        # Fallback: single trivial square
        result = [[(i + j) % 10 for j in range(10)] for i in range(10)]
    
    # Ensure proper types
    result = [[list(map(int, row)) for row in sq] for sq in result]
    
    return result
# EVOLVE-BLOCK-END
