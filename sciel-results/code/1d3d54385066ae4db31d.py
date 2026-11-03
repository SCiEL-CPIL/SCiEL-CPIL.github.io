        for t in range(rank):
            if time.time() > DEADLINE - 15:
                break
            if t in removed:
                continue
            if attempts >= max_attempts:
                break
            
            attempts += 1
            # Try removing term t
            mask = [i for i in range(rank) if i != t and i not in removed]
            if len(mask) < 2:
                continue
            
            U_test = U[:, mask]
            V_test = V[:, mask]
            W_test = W[:, mask]
            
            if verify_decomposition(U_test, V_test, W_test):
                removed.add(t)
