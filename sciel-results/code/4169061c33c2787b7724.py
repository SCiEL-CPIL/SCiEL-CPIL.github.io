    def local_search(M, n, time_limit):
        """
        Fast local search using NumPy for delta calculation.
        Returns True if no mono triangles remain.
        """
        current_count = count_mono_triangles_vectorized(M, n)
        
        if current_count == 0:
            return True
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        no_improve = 0
        best_count = current_count
        
        while time.time() < time_limit:
            if current_count == 0:
                return True
            
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
                
            old_c = M[i, j]
            
            # Calculate counts for each color c:
            # Number of k such that M[i,k]==c and M[j,k]==c
            # We sum over all k, then correct for k=i and k=j if c==0
            counts = np.zeros(4, dtype=np.int32)
            for c in range(4):
                cnt = np.sum((M[i, :] == c) & (M[j, :] == c))
                if c == 0:
                    # k=i: M[i,i]=0. If M[j,i]==0, it was counted.
                    if M[j, i] == 0:
                        cnt -= 1
                    # k=j: M[j,j]=0. If M[i,j]==0, it was counted.
                    if M[i, j] == 0:
                        cnt -= 1
                counts[c] = cnt
            
            best_delta = 0
            best_new_c = old_c
            
            for new_c in range(4):
                if new_c == old_c:
                    continue
                delta = counts[new_c] - counts[old_c]
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = new_c
            
            if best_delta <= 0:
                accept = True
            else:
                if T > min_T:
                    accept = random.random() < np.exp(-best_delta / T)
                else:
                    accept = False
            
            if accept and best_new_c != old_c:
                M[i, j] = best_new_c
                M[j, i] = best_new_c
                current_count += best_delta
                
                if current_count < best_count:
                    best_count = current_count
                    no_improve = 0
                else:
                    no_improve += 1
                
                if current_count == 0:
                    return True
            else:
                no_improve += 1
            
            T *= cooling
            
            # Restart if stuck too long
            if no_improve > 2000:
                # Try a random restart with perturbation
                for _ in range(10):
                    i_r = random.randint(0, n - 1)
                    j_r = random.randint(0, n - 1)
                    if i_r != j_r:
                        M[i_r, j_r] = random.randint(0, 3)
                        M[j_r, i_r] = M[i_r, j_r]
                current_count = count_mono_triangles_vectorized(M, n)
                no_improve = 0
                T = 1.0
        
        return current_count == 0
    