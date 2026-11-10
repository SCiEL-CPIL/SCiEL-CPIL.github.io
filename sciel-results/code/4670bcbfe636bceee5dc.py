def _get_pair_data(S):
    n = len(S)
    if n < 2:
        return None, None, None
    pair_sums = []
    pair_i = []
    pair_j = []
    for i in range(n):
        for j in range(i+1, n):
            pair_sums.append(S[i] + S[j])
            pair_i.append(i)
            pair_j.append(j)
    return np.array(pair_sums), np.array(pair_i), np.array(pair_j)

def _check_validity_with_data(S, x, triple_bits, pair_data):
    if x in S:
        return False
    if triple_bits & _masks[x]:
        return False
    
    n = len(S)
    if n < 2:
        return True
        
    if pair_data is None:
        return True
        
    pair_sums_arr, pair_i_arr, pair_j_arr = pair_data
    
    for idx_a, a in enumerate(S):
        mask = ~((pair_i_arr == idx_a) | (pair_j_arr == idx_a))
        if not np.any(mask):
            continue
        valid_sums = pair_sums_arr[mask]
        if np.any((x + valid_sums) % a == 0):
            return False
            
    return True
