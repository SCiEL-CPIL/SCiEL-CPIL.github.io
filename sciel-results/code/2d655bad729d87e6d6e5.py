    def generate_vectors(weight):
        vecs = []
        for indices in itertools.combinations(range(D), weight):
            for signs in itertools.product([1, -1], repeat=weight):
                v = [0] * D
                for idx, s in zip(indices, signs):
                    v[idx] = s
                vecs.append(v)
        return np.array(vecs, dtype=np.int8)
    