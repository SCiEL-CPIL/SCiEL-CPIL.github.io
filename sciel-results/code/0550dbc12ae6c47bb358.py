            d_left = np.maximum(x[:, 0] + 0.5, 0.05)
            d_right = np.maximum(0.5 - x[:, 0], 0.05)
            d_bottom = np.maximum(x[:, 1] + 0.5, 0.05)
            d_top = np.maximum(0.5 - x[:, 1], 0.05)
