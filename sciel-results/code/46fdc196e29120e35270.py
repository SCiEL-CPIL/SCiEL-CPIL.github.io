    def fitness(pop):
        # pop: P x M
        cost = 0
        # 4-sets
        # pop[:, idx_4] is P x N_4 x 4
        # sum along axis 2
        sums4 = np.sum(pop[:, idx_4], axis=2)
        cost += np.sum(sums4 == 0)
        # 5-sets
        sums5 = np.sum(pop[:, idx_5], axis=2)
        cost += np.sum(sums5 == 10)
        return cost
    