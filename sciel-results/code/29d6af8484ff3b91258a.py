# EVOLVE-BLOCK-START
import time

N = 256

def construct():
    start_time = time.time()
    
    # 1. Greedy initialization (same as seed)
    selected = []
    in_S = [False] * (N + 1)
    
    for x in range(1, N + 1):
        valid = True
        for a in selected:
            for b in selected:
                if a < b:
                    if 2 * b - a == x:
                        valid = False
                        break
            if not valid:
                break
        if valid:
            selected.append(x)
            in_S[x] = True
            
    S = set(selected)
    
    # Helper to check if z can be added to S_temp (which is S without y)
    # z is valid if no a, b in S_temp form a 3-AP with z
    # Cases:
    # 1. z is middle: a, z, b => 2z = a+b => b = 2z-a. Check if b in S_temp and b != a.
    #    Also need a < z < b or b < z < a.
    # 2. z is largest: a, b, z => 2b = a+z => b = (a+z)/2. Check if b in S_temp and a < b < z.
    # 3. z is smallest: z, a, b => 2a = z+b => b = 2a-z. Check if b in S_temp and z < a < b.
    
    def is_compatible(z, S_temp, in_S_temp):
        # in_S_temp is a boolean array for S_temp for O(1) lookups
        # We iterate through a in S_temp
        for a in S_temp:
            # Case 1: z is middle. a, z, b or b, z, a
            # b = 2z - a
            b = 2 * z - a
            if b != a and 1 <= b <= N and b in S_temp:
                # Check if a, z, b is a non-constant AP
                # Since b != a, difference is non-zero.
                # We just need to ensure it's a valid AP.
                # The condition 2z = a+b is sufficient.
                return False
            
            # Case 2: z is largest. a, b, z
            # b = (a + z) / 2
            if (a + z) % 2 == 0:
                b = (a + z) // 2
                if b != a and b != z and 1 <= b <= N and b in S_temp:
                    # Ensure a < b < z for a, b, z
                    # Or b < a < z?
                    # If a, b, z is AP, then 2b = a+z.
                    # If a < z, then b is between them.
                    # So a < b < z or z < b < a.
                    # Since z is largest in this case assumption?
                    # No, we just check if {a, b, z} is an AP.
                    # If b in S_temp and 2b == a+z, then {a,b,z} is AP.
                    return False

            # Case 3: z is smallest. z, a, b
            # b = 2a - z
            b = 2 * a - z
            if b != a and b != z and 1 <= b <= N and b in S_temp:
                # Check if {z, a, b} is AP.
                # 2a = z+b.
                return False
        
        return True

    improved = True
    while improved:
        if time.time() - start_time > 80:
            break
        improved = False
        # Try to swap larger elements with smaller ones
        # Iterate y in S in descending order
        for y in sorted(S, reverse=True):
            # Create temporary set without y
            S_temp = S - {y}
            in_S_temp = in_S[:]
            in_S_temp[y] = False
            
            # Try to add z < y not in S
            for z in range(1, y):
                if z in S:
                    continue
                
                # Check compatibility of z with S_temp
                if is_compatible(z, S_temp, in_S_temp):
                    # Perform swap
                    S.remove(y)
                    S.add(z)
                    in_S[y] = False
                    in_S[z] = True
                    improved = True
                    break
            if improved:
                break
                
    return sorted(S)
# EVOLVE-BLOCK-END
