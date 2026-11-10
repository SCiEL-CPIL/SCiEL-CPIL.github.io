        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        for i in range(n_samples):
            for j in range(self.dim):
                x_samples[i, j] = np.clip(x_samples[i, j], lo[j], hi[j])
