<<<<<<< SEARCH
x0_values = np.zeros(self.budget)
   x0_values[0] = 0.0
   x0_values[-1] = 0.15
   if self.budget > 2:
       x0_values[1:-1] = np.linspace(0.1, 0.15, self.budget - 2)
=======
x0_values = np.zeros(self.budget)
x0_values[0] = 0.0
x0_values[-1] = 0.15
if self.budget > 2:
    x0_values[1:-1] = np.linspace(0.1, 0.15, self.budget - 2)
>>>>>>> REPLACE
