        prev_in_S = in_S[:]
        prev_size = S_size
        
        # ... kick and refill ...
        
        delta = S_size - prev_size
        if delta < 0:
            prob = 2.718281828 ** (delta / T)
            if random.random() >= prob:
                # Reject
                in_S[:] = prev_in_S
                S_size = prev_size
