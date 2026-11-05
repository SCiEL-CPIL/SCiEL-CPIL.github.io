<<<<<<< SEARCH
    if n_nodes > 2:
        # d_feas_feas already computed above (with inf diagonal)
        # For 2-step: for each i, soft_min over j of (d(i,j) + cont_cost[j])
        two_step = np.zeros(n_nodes)
        for i in range(n_nodes):
            # Options: for each j != i, cost = d(i,j) + cont_cost[j]
            costs_j = d_feas_feas[i, :] + cont_cost
            costs_j_safe = np.where(np.isinf(costs_j), 1e6, costs_j)
            neg_cs = -costs_j_safe / tau
            max_neg_cs = np.max(neg_cs)
            lse_cs = max_neg_cs + np.log(np.sum(np.exp(neg_cs - max_neg_cs)))
            two_step[i] = -tau * lse_cs
    else:
        two_step = cont_cost  # fallback
=======
    if n_nodes > 2:
        # d_feas_feas already computed above (with inf diagonal)
        # For 2-step: for each i, soft_min over j of (d(i,j) + cont_cost[j])
        # Vectorized: costs_matrix[i,j] = d(i,j) + cont_cost[j]
        costs_matrix = d_feas_feas + cont_cost.reshape(1, -1)
        costs_matrix_safe = np.where(np.isinf(costs_matrix), 1e6, costs_matrix)
        neg_cs = -costs_matrix_safe / tau
        max_neg_cs = np.max(neg_cs, axis=1, keepdims=True)
        lse_cs = max_neg_cs + np.log(np.sum(np.exp(neg_cs - max_neg_cs), axis=1, keepdims=True))
        two_step = -tau * lse_cs.squeeze()
    else:
        two_step = cont_cost  # fallback
>>>>>>> REPLACE
