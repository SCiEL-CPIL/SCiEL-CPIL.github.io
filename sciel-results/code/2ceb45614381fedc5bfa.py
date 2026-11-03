    def local_search(U, V, W, deadline, rng):
        r = U.shape[1]
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            indices = list(range(r))
            rng.shuffle(indices)
            for l in indices:
                if time.time() > deadline - 10:
                    break
                if try_drop(U, V, W, l):
                    U = np.delete(U, l, axis=1)
                    V = np.delete(V, l, axis=1)
                    W = np.delete(W, l, axis=1)
                    r -= 1
                    improved = True
                    break
        return U, V, W
    