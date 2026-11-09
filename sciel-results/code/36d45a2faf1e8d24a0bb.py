import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Bit i represents element i+1
    # U = [65,128] corresponds to bits 64..127
    # L = [1,64] corresponds to bits 0..63
    
    def check_add_bit(x, L_mask, U_mask, S_mask):
        """Check if adding element x (1-indexed) is safe. Returns True if safe."""
        # A_x: elements a in L such that x+a in S
        # a = i+1, x+a = x+i+1, bit index = x+i
        # Bit i in (S_mask >> x) means element i+1+x is in S
        # So A_x = L_mask & (S_mask >> x)
        A_x = L_mask & (S_mask >> x)
        
        if A_x == 0:
            return True  # No potential conflicts
        
        # Type A: check if there exist a, b in A_x, a < b, with a+b in S
        # a = i+1, b = j+1, i < j
        # a+b = i+j+2, bit index = i+j+1
        temp = A_x
        while temp:
            lowbit = temp & (-temp)
            i = lowbit.bit_length() - 1
            a = i + 1
            # For this a, check if there exists b > a in A_x with a+b in S
            # b = j+1 > a = i+1 means j > i
            # a+b = i+j+2, need bit (i+j+1) in S
            # (S_mask >> (i+1)) has bit j set iff element j+1+i+1 = j+i+2 is in S
            # So (A_x & (S_mask >> (i+1))) has bit j set iff element j+1 in A_x and element j+i+2 in S
            # We need j > i, so mask out bits 0..i
            T = A_x & (S_mask >> (i + 1))
            if T & ~((1 << (i + 1)) - 1):
                return False
            temp &= temp - 1
        
        # Type B: check if there exist a in A_x, c in U, with x+c in S and a+c in S
        # C_x = {c in U : x+c in S} = U_mask & (S_mask >> x)
        C_x = U_mask & (S_mask >> x)
        
        if C_x == 0:
            return True
        
        temp = A_x
        while temp:
            lowbit = temp & (-temp)
            i = lowbit.bit_length() - 1
            a = i + 1
            # C_a = {c in U : a+c in S} = U_mask & (S_mask >> a)
            C_a = U_mask & (S_mask >> a)
            if C_x & C_a:
                return False
            temp &= temp - 1
        
        return True
    
    # Initial state
    U_mask = ((1 << 128) - 1) ^ ((1 << 64) - 1)  # bits 64..127 set
    L_mask = 0
    S_mask = U_mask
    
    all_L_mask = (1 << 64) - 1  # bits 0..63
    
    best_size = 64
    best_L_mask = 0
    best_U_mask = U_mask
    
    def branch_and_bound(remaining_mask, L_cur, U_cur, S_cur, size):
        nonlocal best_size, best_L_mask, best_U_mask, start_time
        
        if time.time() - start_time > DEADLINE:
            return
        
        # Upper bound
        if size + remaining_mask.bit_count() <= best_size:
            return
        
        if remaining_mask == 0:
            if size > best_size:
                best_size = size
                best_L_mask = L_cur
                best_U_mask = U_cur
            return
        
        # Greedy upper bound
        safe_count = 0
        temp = remaining_mask
        while temp:
            lowbit = temp & (-temp)
            i = lowbit.bit_length() - 1
            x = i + 1
            if check_add_bit(x, L_cur, U_cur, S_cur):
                safe_count += 1
            temp &= temp - 1
        
        if size + safe_count <= best_size:
            return
        
        # Process elements
        temp = remaining_mask
        while temp:
            lowbit = temp & (-temp)
            i = lowbit.bit_length() - 1
            x = i + 1
            bit = 1 << i
            remaining_without = remaining_mask & ~bit
            
            if time.time() - start_time > DEADLINE:
                return
            
            # Try include first if safe
            if check_add_bit(x, L_cur, U_cur, S_cur):
                branch_and_bound(remaining_without, L_cur | bit, U_cur, S_cur | bit, size + 1)
            
            # Try exclude
            branch_and_bound(remaining_without, L_cur, U_cur, S_cur, size)
            
            if time.time() - start_time > DEADLINE:
                return
            
            temp &= temp - 1
    
    # Run B&B
    branch_and_bound(all_L_mask, 0, U_mask, U_mask, 64)
    
    # Local search polish
    if time.time() - start_time < DEADLINE - 5:
        L_cur = best_L_mask
        U_cur = best_U_mask
        S_cur = L_cur | U_cur
        
        improved = True
        while improved and time.time() - start_time < DEADLINE - 2:
            improved = False
            
            # Try to add any L element
            for x in range(1, 65):
                bit = 1 << (x - 1)
                if S_cur & bit:
                    continue
                if check_add_bit(x, L_cur, U_cur, S_cur):
                    L_cur |= bit
                    S_cur |= bit
                    improved = True
                    break
            
            if not improved:
                # Try removing one U element
                for u in range(65, 129):
                    if time.time() - start_time > DEADLINE - 2:
                        break
                    ubit = 1 << (u - 1)
                    if not (U_cur & ubit):
                        continue
                    new_U = U_cur & ~ubit
                    new_L = L_cur
                    new_S = S_cur & ~ubit
                    added = True
                    while added and time.time() - start_time < DEADLINE - 2:
                        added = False
                        for x in range(1, 65):
                            xbit = 1 << (x - 1)
                            if new_S & xbit:
                                continue
                            if check_add_bit(x, new_L, new_U, new_S):
                                new_L |= xbit
                                new_S |= xbit
                                added = True
                                break
                    if new_S.bit_count() > S_cur.bit_count():
                        L_cur = new_L
                        U_cur = new_U
                        S_cur = new_S
                        improved = True
                        break
    
    # Convert bitmask to list
    result = []
    for i in range(128):
        if S_cur & (1 << i):
            result.append(i + 1)
    
    result = [int(x) for x in result]
    return result

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Set: {result[:20]}...{result[-20:]}")
    
    # Verify
    S = set(result)
    valid = True
    for a in result:
        for b in result:
            if a >= b:
                continue
            for c in result:
                if c <= b:
                    continue
                if (a+b) in S and (a+c) in S and (b+c) in S:
                    valid = False
                    print(f"Forbidden triple: {a},{b},{c}")
                    break
            if not valid:
                break
        if not valid:
            break
    print(f"Valid: {valid}")
