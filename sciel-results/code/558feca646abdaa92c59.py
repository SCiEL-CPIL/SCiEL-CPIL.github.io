    code_list = list(code_set)
    used = np.zeros(2**17, dtype=bool)
    for w in code_list:
        used[w] = True
        
    code_arr = np.array(code_list, dtype=np.uint32)
    
    # Try to add words
    # Iterate through all words
    for w in range(2**17):
        if time.time() > deadline - 5:
            break
        if used[w]:
            continue
            
        w_np = np.uint32(w)
        dists = np.bit_count(code_arr ^ w_np)
        if np.min(dists) >= 6:
            code_list.append(w)
            code_arr = np.append(code_arr, w)
            used[w] = True
            
    return code_list
