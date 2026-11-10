    def project_to_sphere(pts):
        """Project points onto unit sphere"""
        # Ensure pts is 2D
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
            was_1d = True
        else:
            was_1d = False
            
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        res = pts / norms
        
        if was_1d:
            return res[0]
        return res
