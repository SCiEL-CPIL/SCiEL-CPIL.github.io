    def objective(v):
        x, y, s = unpack(v)
        
        pen = 0.0
        for i in range(n):
            # x bounds
            pen += max(0.0, s[i]/2 - x[i])**2
            pen += max(0.0, x[i] - (1.0 - s[i]/2))**2
            # y bounds
            pen += max(0.0, s[i]/2 - y[i])**2
            pen += max(0.0, y[i] - (1.0 - s[i]/2))**2
            
        for i in range(n):
            for j in range(i+1, n):
                dx = np.sqrt((x[i] - x[j])**2 + 1e-12)
                dy = np.sqrt((y[i] - y[j])**2 + 1e-12)
                r = (s[i] + s[j]) / 2
                pen_x = max(0.0, r - dx)
                pen_y = max(0.0, r - dy)
                pen += pen_x**2 + pen_y**2
                
        return -np.sum(s) + 100.0 * pen
