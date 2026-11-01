    def try_flip(h, A, i):
        h_i = h[i]
        delta = np.zeros(N)
        # k > 0
        # Term 1: k in [1, N-1-i]
        # h[i+k]
        if N - i - 1 >= 1:
            delta[1:N-i] -= 2 * h_i * h[i+1:N]
        # Term 2: k in [1, i]
        # h[i-k]
        if i >= 1:
            # k=1 -> h[i-1]
            # k=i -> h[0]
            # h[0:i][::-1] is h[i-1], ..., h[0]
            delta[1:i+1] -= 2 * h_i * h[0:i][::-1]
        
        # k=0 is unchanged
        new_A = A + delta
        new_val = np.max(new_A)
        return new_A, new_val
