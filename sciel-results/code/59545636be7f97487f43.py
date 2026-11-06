<<<<<<< SEARCH
    for i in range(num_powerful):
        a = powerful_list[i]
        # Early break if 2*a > max_possible_N? 
        # We don't have a fixed max N, but N = a+b.
        # If we want to find the *best* N, we should consider all sums.
        
        for j in range(i, num_powerful):
            b = powerful_list[j]
            N = a + b
            rep_count[N] += 1
            
        # Time check
        if time.time() > deadline - 10:
            break
=======
    try:
        for i in range(num_powerful):
            a = powerful_list[i]
            for j in range(i, num_powerful):
                b = powerful_list[j]
                N = a + b
                rep_count[N] += 1
            
            # Time check
            if time.time() > deadline - 10:
                break
    except MemoryError:
        pass
>>>>>>> REPLACE
