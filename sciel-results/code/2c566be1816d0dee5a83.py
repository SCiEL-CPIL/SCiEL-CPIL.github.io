"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import math
    import time
    
    start_time = time.time()
    DEADLINE = 50.0  # Safety margin
    
    N = 512
    odds = [i for i in range(1, N + 1) if i % 2 == 1]
    
    # State: list of (odd_part, max_power_2)
    selected = []
    
    # Sort odds by value? Smaller odds allow higher powers.
    # Adding smaller odds first might constrain larger ones more?
    # Or larger odds first?
    # Let's try sorting by odd value ascending.
    
    for k in odds:
        if time.time() > DEADLINE:
            break
            
        # Calculate max possible power for k based on itself
        # k * 2^h <= 512
        max_h_self = 0
        temp = k
        while temp <= N:
            max_h_self += 1
            temp *= 2
        max_h_self -= 1 # h is exponent, count was h+1
        
        # Check compatibility with existing selected
        # For each existing (j, hj), calculate M_j = floor(log2(512 / lcm(k, j)))
        # Condition: if we pick h_k = h, then for all j:
        # if h >= hj, h <= M_j
        # if h < hj, hj <= M_j (must hold for feasibility)
        
        feasible = True
        min_h_if_high = max_h_self
        
        for j, hj in selected:
            # Calculate LCM of odd parts
            g = math.gcd(k, j)
            lcm_kj = (k // g) * j
            
            # M_j is max power allowed if h_k >= hj
            # 2^M_j * lcm_kj <= 512
            m_j = 0
            temp_lcm = lcm_kj
            while temp_lcm <= N:
                m_j += 1
                temp_lcm *= 2
            m_j -= 1
            
            # If hj > m_j, we MUST have h_k >= hj AND h_k <= m_j.
            # This is impossible if hj > m_j.
            if hj > m_j:
                feasible = False
                break
            
            # If h_k >= hj, we need h_k <= m_j.
            # So if we choose h_k >= hj, h_k is capped by m_j.
            # We don't know h_k yet.
            # But we know that if we pick h_k >= hj, h_k cannot exceed m_j.
            # This effectively caps h_k if h_k ends up being >= hj.
            
            # We can refine the upper bound for h_k.
            # If we plan to pick h_k >= hj, then h_k <= m_j.
            # If we plan to pick h_k < hj, no cap from this j (since hj <= m_j is already checked).
            
            # To find max h_k, we can just iterate down from max_h_self.
            # But we can tighten the bound.
            # If max_h_self >= hj, then any candidate h >= hj must be <= m_j.
            # So effective max is min(max_h_self, m_j) if we assume h >= hj?
            # It's complex to pre-compute. Iteration is fast (max 9 steps).
            pass
        
        if not feasible:
            continue
            
        # Find max h
        found_h = -1
        for h in range(max_h_self, -1, -1):
            valid = True
            for j, hj in selected:
                g = math.gcd(k, j)
                lcm_kj = (k // g) * j
                
                # Check 2^max(h, hj) * lcm_kj <= 512
                max_pow = max(h, hj)
                if lcm_kj * (1 << max_pow) > N:
                    valid = False
                    break
            if valid:
                found_h = h
                break
        
        if found_h >= 0:
            selected.append((k, found_h))
            
    # Construct the final set
    result = set()
    for k, h in selected:
        for v in range(h + 1):
            result.add(k * (1 << v))
            
    return list(result)

# EVOLVE-BLOCK-END
