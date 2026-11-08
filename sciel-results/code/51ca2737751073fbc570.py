    class DecompBuilder:
        def __init__(self, n, m, p):
            self.n = n
            self.m = m
            self.p = p
            self.rank = 0
            self.U = []
            self.V = []
            self.W = []
            
        def add_term(self, u_vec, v_vec, w_vec):
            # u_vec: list of length n*m
            # v_vec: list of length m*p
            # w_vec: list of length p*n
            col = self.rank
            self.rank += 1
            self.U.append(u_vec)
            self.V.append(v_vec)
            self.W.append(w_vec)
            
        def get_matrices(self):
            # Transpose to (dim, rank)
            U = [[0]*self.rank for _ in range(self.n*self.m)]
            V = [[0]*self.rank for _ in range(self.m*self.p)]
            W = [[0]*self.rank for _ in range(self.p*self.n)]
            for r in range(self.rank):
                for i in range(self.n*self.m):
                    U[i][r] = self.U[r][i]
                for j in range(self.m*self.p):
                    V[j][r] = self.V[r][j]
                for k in range(self.p*self.n):
                    W[k][r] = self.W[r][k]
            return U, V, W
    