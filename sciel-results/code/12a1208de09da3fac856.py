   def design(problem):
       size = problem["pixels"]
       cell = []
       half_size = size / 2.0
       # Side length 0.35 of lattice constant
       half_side = 0.35 / 2.0
       
       for i in range(size):
           row = []
           y = (i + 0.5) / size - 0.5
           for j in range(size):
               x = (j + 0.5) / size - 0.5
               # Inside the square hole?
               if abs(x) < half_side and abs(y) < half_side:
                   row.append(0) # Air
               else:
                   row.append(1) # Dielectric
           cell.append(row)
       return cell
   