class MatMulResult:
    def __init__(self, U, V, W):
        self.U = U
        self.V = V
        self.W = W
        
    def __getitem__(self, key):
        if key == 'U': return self.U
        if key == 'V': return self.V
        if key == 'W': return self.W
        raise KeyError(key)
