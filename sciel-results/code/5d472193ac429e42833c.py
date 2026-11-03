<<<<<<< SEARCH
def _valid_voltage_set(v, m, g):
    """Check if a voltage set has the required properties for girth >= g."""
    v = [x % m for x in v]
    if len(set(v)) != 4 or 0 in v:
        return False
    
    # Check no 4-cycles: all pair sums (v_i + v_j) mod m must be distinct for i < j
    pair_sums = set()
    for i in range(4):
        for j in range(i + 1, 4):
            s = (v[i] + v[j]) % m
            if s in pair_sums:
                return False
            pair_sums.add(s)
    
    # Check no 6-cycles: no ±v_i ± v_j ± v_k = 0 mod m for distinct i,j,k
    # This means: v_i + v_j + v_k != 0, v_i + v_j - v_k != 0, v_i - v_j + v_k != 0, v_i - v_j - v_k != 0
    for i in range(4):
        for j in range(i + 1, 4):
            for k in range(j + 1, 4):
                if (v[i] + v[j] + v[k]) % m == 0:
                    return False
                if (v[i] + v[j] - v[k]) % m == 0:
                    return False
                if (v[i] - v[j] + v[k]) % m == 0:
                    return False
                if (v[i] - v[j] - v[k]) % m == 0:
                    return False
    
    # Check no 8-cycles: no ±v_i ± v_j ± v_k ± v_l = 0 mod m for distinct i,j,k,l
    # With 4 elements, we need to check all sign combinations
    # v_i + v_j + v_k + v_l
    # v_i + v_j + v_k - v_l
    # v_i + v_j - v_k + v_l
    # v_i + v_j - v_k - v_l
    # v_i - v_j + v_k + v_l
    # v_i - v_j + v_k - v_l
    # v_i - v_j - v_k + v_l
    # v_i - v_j - v_k - v_l
    # But by symmetry, we only need to check a subset
    
    vi, vj, vk, vl = v[0], v[1], v[2], v[3]
    if (vi + vj + vk + vl) % m == 0:
        return False
    if (vi + vj + vk - vl) % m == 0:
        return False
    if (vi + vj - vk + vl) % m == 0:
        return False
    if (vi + vj - vk - vl) % m == 0:
        return False
    if (vi - vj + vk + vl) % m == 0:
        return False
    if (vi - vj + vk - vl) % m == 0:
        return False
    if (vi - vj - vk + vl) % m == 0:
        return False
    if (vi - vj - vk - vl) % m == 0:
        return False
    
    return True
=======
def _valid_voltage_set(v, m, g):
    """Check if a voltage set has the required properties for girth >= g."""
    v = [x % m for x in v]
    if len(set(v)) != 4 or 0 in v:
        return False
    
    # Check no 4-cycles: all pair sums (v_i + v_j) mod m must be distinct for i < j
    pair_sums = set()
    for i in range(4):
        for j in range(i + 1, 4):
            s = (v[i] + v[j]) % m
            if s in pair_sums:
                return False
            pair_sums.add(s)
    
    # Check no 6-cycles: no ±v_i ± v_j ± v_k = 0 mod m for distinct i,j,k
    for i in range(4):
        for j in range(i + 1, 4):
            for k in range(j + 1, 4):
                if (v[i] + v[j] + v[k]) % m == 0:
                    return False
                if (v[i] + v[j] - v[k]) % m == 0:
                    return False
                if (v[i] - v[j] + v[k]) % m == 0:
                    return False
                if (v[i] - v[j] - v[k]) % m == 0:
                    return False
    
    # Check no 8-cycles: no ±v_i ± v_j ± v_k ± v_l = 0 mod m
    vi, vj, vk, vl = v[0], v[1], v[2], v[3]
    if (vi + vj + vk + vl) % m == 0:
        return False
    if (vi + vj + vk - vl) % m == 0:
        return False
    if (vi + vj - vk + vl) % m == 0:
        return False
    if (vi + vj - vk - vl) % m == 0:
        return False
    if (vi - vj + vk + vl) % m == 0:
        return False
    if (vi - vj + vk - vl) % m == 0:
        return False
    if (vi - vj - vk + vl) % m == 0:
        return False
    if (vi - vj - vk - vl) % m == 0:
        return False
    
    return True
>>>>>>> REPLACE

