<<<<<<< SEARCH
    # Convert to list and apply local search
    selected_list = list(selected)
    if time.time() < DEADLINE - 5:
        selected_list = _local_search(q, n, radius, total, neighbors, selected_list, DEADLINE)
    
    # Convert indices to codewords
    result = []
    for idx in selected_list:
        digits = [0] * n
        temp = idx
        for k in range(n):
            digits[k] = temp % q
            temp //= q
        result.append(digits)
    
    return result
=======
    # Convert to list and apply local search
    selected_list = list(selected)
    greedy_backup = list(selected_list) # Keep a valid backup
    
    if time.time() < DEADLINE - 5:
        selected_list = _local_search(q, n, radius, total, neighbors, selected_list, DEADLINE)
        
        # Validate the result of local search
        if not selected_list or not _is_valid_covering_code(q, n, radius, total, neighbors, selected_list):
            selected_list = greedy_backup

    # Convert indices to codewords
    result = []
    for idx in selected_list:
        digits = [0] * n
        temp = idx
        for k in range(n):
            digits[k] = temp % q
            temp //= q
        result.append(digits)
    
    return result
>>>>>>> REPLACE
