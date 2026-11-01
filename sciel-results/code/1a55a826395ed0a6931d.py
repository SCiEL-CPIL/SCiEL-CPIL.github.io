        xy[:, 0] = np.clip(xy[:, 0], r, 1.0 - r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1.0 - r)
        final_params = np.concatenate([xy.flatten(), r])
        if is_valid(final_params):
